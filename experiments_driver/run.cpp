#include "util.h"
#include "../hnswlib/shiro_ef.h"
#include <filesystem>
#include <cstdlib>

// ============================================================================
// GLOBAL CONFIGURATION
// Easily configure parameters here instead of modifying them in each function.
// ============================================================================
struct ExperimentConfig {
    std::string dataset;
    std::string metric;
    size_t k;
    float alpha;
    float gamma;
    float expected_recall;
    int ef_upper_bound;
    int repeat;
    int sampling_size;
    int n_cv_tables;
    int min_queries_per_score;
    size_t statics_length;
};

static std::vector<ExperimentConfig> g_experiments = {
    // dataset, metric, k, alpha, gamma, expected_recall, ef_upper_bound, repeat, sampling_size, n_cv_tables, min_q, statics_length
    {"deep-image-96-angular",      "cd", 100, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    {"glove-100-angular",          "cd", 100, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    {"sift-128-euclidean",         "l2", 100, 0.25f, 12.0f, 0.95f,  300, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"msmarco",                 "cd", 1000, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"cohere",                  "cd", 1000, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"laion_image",             "cd", 1000, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"laion_text",              "cd", 1000, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"cluster_mg_uniform_100d", "cd", 1000, 0.251f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32},
    // {"cluster_mg_zipf_100d",    "cd", 1000, 0.25f, 12.0f, 0.95f, 5000, 3, 3000, 15, 3, 1 + 32 + 31 * 32}
};

static ExperimentConfig get_config(const std::string& dataset) {
    for (const auto& conf : g_experiments) {
        if (conf.dataset == dataset) return conf;
    }
    return g_experiments[0]; // fallback
}
// ============================================================================



static hnswdis::Sketch make_sketch(const hnswdis::EfAdapter &adapter, float expected_recall)
{
    if (adapter.has_cv_tables())
        return hnswdis::Sketch(adapter.get_all_tables(), adapter.get_cv_centers(), expected_recall);
    return hnswdis::Sketch(adapter.get_ef_recall_estimators(), expected_recall);
}

static void train_cv_buckets(
    hnswdis::EfAdapter &adapter,
    const std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw,
    const std::shared_ptr<hnswdis::MatrixXf> data,
    const size_t k,
    const std::string &metric,
    const float alpha,
    const float gamma,
    const size_t statics_length,
    const std::shared_ptr<hnswdis::MatrixXf> query_vectors,
    const std::shared_ptr<hnswdis::MatrixXi> ground_truth,
    const int ef_upper_bound,
    const int n_cv_tables,
    const int min_queries_per_score,
    const std::string &samplings_path = "")
{
    adapter.init_with_cv_buckets(
        hnsw, data, k, metric, alpha, gamma, statics_length,
        query_vectors, ground_truth,
        n_cv_tables, min_queries_per_score);

    if (samplings_path != "") {
        size_t num_hard_queries = 0;
        std::ifstream meta_in(samplings_path + ".meta");
        if (meta_in.good()) {
            meta_in >> num_hard_queries;
        }

        if (num_hard_queries > 0 && num_hard_queries <= query_vectors->rows()) {
            float ef_hard_sum = 0;
            float ef_easy_sum = 0;
            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
            hnswdis::Sketch temp_sketch(adapter.get_all_tables(), adapter.get_cv_centers(), adapter.get_expected_recall());

            for(size_t i = 0; i < query_vectors->rows(); ++i) {
                float cv = 0.0f;
                auto ret = hnsw->adaptiveSearchKnn(query_vectors->row(i).data(), k, statics_length, score_cal, nullptr, &cv);
                int cv_score = std::max(0, std::min(100, static_cast<int>(cv * 500.0f)));
                size_t est_ef = temp_sketch.estimate_ef2(cv_score, ret.second);
                if (i < num_hard_queries) ef_hard_sum += est_ef;
                else ef_easy_sum += est_ef;
            }

            float ef_hard = ef_hard_sum / num_hard_queries;
            float ef_easy = (query_vectors->rows() - num_hard_queries) > 0 ? ef_easy_sum / (query_vectors->rows() - num_hard_queries) : 0;
            float hard_pct = static_cast<float>(num_hard_queries) / 30000.0f;
            float true_wae = hard_pct * ef_hard + (1.0f - hard_pct) * ef_easy;

            adapter.set_wae(true_wae);
            std::cout << "Reconstructed True WAE: " << true_wae << " (Hard Pct: " << hard_pct << ")" << std::endl;
        }
    }
}

const char *experiments_root = std::getenv("EXPERIMENTS_ROOT");
const auto root = experiments_root ? std::filesystem::path(experiments_root)
                                   : std::filesystem::current_path();

void setup_laion_text2image(std::shared_ptr<hnswlib::HierarchicalNSW<float>> &hnsw,
                            std::shared_ptr<hnswdis::MatrixXf> &query,
                            std::shared_ptr<hnswdis::MatrixXf> &data,
                            std::shared_ptr<hnswdis::MatrixXi> &ground_truth,
                            std::shared_ptr<hnswlib::SpaceInterface<float>> &space)
{
    // laion_text_ng.hdf contains the full corpus of laion_text and also the query set
    // laion_text_query_groundtruth.hdf contains the query set and the ground truth that are the images in the laion_image corpus
    // in the case of text to image retrieval, the query set is the laion_text corpus, the data set and the ground truth are the laion_image corpus

    std::string index_path = (root / "index/laion_image-M16-efc-500-parallel.hnsw").string();
    std::string laion_image_path = (root / "data/laion_image.hdf5").string();
    std::string laion_text_path = (root / "data/laion_text_query_groundtruth.hdf5").string();

    hnswdis::MatrixXf image_data;
    hnswdis::MatrixXf text_query;
    hnswdis::MatrixXi image_ground_truth;

    load_hdf5(laion_image_path, "train", image_data);                                // load the image data
    load_hdf5(laion_text_path, "test", text_query, "neighbors", image_ground_truth); // load the text query and the ground truth

    std::cout << "Normalize the data vectors" << std::endl;
    normalize_matrix(image_data);
    normalize_matrix(text_query);

    std::cout << "Data vectors dimensions: " << image_data.rows() << " x " << image_data.cols() << std::endl;
    std::cout << "Query vectors dimensions: " << text_query.rows() << " x " << text_query.cols() << std::endl;
    std::cout << "Neighbors dimensions: " << image_ground_truth.rows() << " x " << image_ground_truth.cols() << std::endl;

    data = std::make_shared<hnswdis::MatrixXf>(image_data);
    query = std::make_shared<hnswdis::MatrixXf>(text_query);
    ground_truth = std::make_shared<hnswdis::MatrixXi>(image_ground_truth);

    // load the index
    space = hnswdis::init_space("cd", text_query.cols());
    hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);

    std::cout << "Index loaded" << std::endl;
    std::cout << "Dimension of space:" << *(size_t *)(space->get_dist_func_param()) << std::endl;
    std::cout << "Data size of space:" << space->get_data_size() << std::endl;
}

void online_exp()
{
    std::cout << "Starting adaptive ef tests...\n\n"
              << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 available threads for eigen parallelization in shiro-ef offline computation

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;
        int repeat = conf.repeat;

        std::cout << "Dataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Truncation ratio: " << alpha << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        if (dataset == "laion_text")
        {
            setup_laion_text2image(hnsw, query, data, ground_truth, space);
        }
        else
        {
            std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
            std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
            auto tuple = load_index_and_data(hdf5_path, index_path, metric);
            hnsw = std::get<0>(tuple);
            query = std::get<1>(tuple);
            data = std::get<2>(tuple);
            ground_truth = std::get<3>(tuple);
            space = std::get<4>(tuple);
        }

        // the followings are for adaptive ef experiments
        std::string ef_adaptor_path = (root / "estimation_table" / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table
        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();           // path for sampling (queries and ground truth)

        if (dataset == "laion_text")
        {
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        // 1. load estimator
        hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

        // 2. load ef_adaptor
        std::shared_ptr<hnswdis::EfAdapter> ef_adapter_ptr;
        hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
        ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);

        // 3. create sketch
        hnswdis::Sketch sketch = make_sketch(*ef_adapter_ptr, expected_recall);
        const float wae = ef_adapter_ptr->get_wae();
        std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
                hnsw->setEf(wae);
        adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        adaptive_ef_analysis(dataset, *hnsw, *query, score_cal, k, sketch, statics_length); // this is used to get the distribution of adaptive ef values

        search_with_patience_in_proximity(dataset, repeat, *hnsw, *query, *ground_truth, k); // hnsw search with various ef values
        baseline_search(dataset, repeat, *hnsw, *query, *ground_truth, k, ef_upper_bound);   // hnsw search with various ef values
    }
}

void indexing_exp()
{
    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;

        std::cout << "Dataset: " << dataset << std::endl;
        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();

        build_index(hdf5_path, index_path, 16, 500, metric, std::max(1u, std::thread::hardware_concurrency() / 4));
    }
}

void offline_laion_text2image()
{
    std::string laion_image_path = (root / "data/laion_image.hdf5").string();
    hnswdis::MatrixXf image_data;
    // load the image data
    load_hdf5(laion_image_path, "train", image_data);
    std::cout << "Normalize the image data vectors" << std::endl;
    normalize_matrix(image_data);

    std::string laion_text_path = (root / "data/laion_text_ng.hdf5").string();
    hnswdis::MatrixXf text_data;
    // load the text data, used for sampling
    load_hdf5(laion_text_path, "train", text_data);
    std::cout << "Normalize the text data vectors" << std::endl;
    normalize_matrix(text_data);

    std::string index_path = (root / "index/laion_image-M16-efc-500-parallel.hnsw").string();
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    // load the index
    auto space = hnswdis::init_space("cd", image_data.cols());
    hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    std::cout << "Index loaded" << std::endl;
    std::cout << "Dimension of space:" << *(size_t *)(space->get_dist_func_param()) << std::endl;
    std::cout << "Data size of space:" << space->get_data_size() << std::endl;

    float expected_recall = 0.95;
    auto conf = get_config("laion_text");
    int sampling_size = conf.sampling_size;
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
    size_t statics_length = conf.statics_length;
    float gamma = conf.gamma;
    float alpha = conf.alpha;
    int ef_upper_bound = conf.ef_upper_bound;

    int k = 1000;

    std::string samplings_path = (root / ("sampling/laion_text-samplings--k" + std::to_string(k) + "-ef.bin")).string();           // path for sampling (queries and ground truth)
    std::string ef_adaptor_path = (root / ("estimation_table/laion_text-ef_adaptor--k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int repeat = 1;

    // 1. compute estimator: this is not required for laion_text as in the case of text-to-image retrieval the data is image so the estimator has alread been computed in the case of laion_image

    // 2. Sample data and compute ground truth
    //  sampling from the text data and compute the ground truth over the image data
    for (int i = 0; i < repeat; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors = hnswdis::sample_data(text_data, sampling_size);
        hnswdis::MatrixXi sample_ground_truth = hnswdis::compute_ground_truth_batch_parallel4(*sample_query_vectors, image_data, "cd", k);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Sampling computing time: " << duration << " ms" << std::endl;
        hnswdis::serialize_samplings(samplings_path, *sample_query_vectors, sample_ground_truth);
    }

    std::shared_ptr<hnswdis::MatrixXf> data = std::make_shared<hnswdis::MatrixXf>(image_data);

    // 3. compute ef_adaptor
    for (int i = 0; i < repeat; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        std::ifstream ef_adaptor_file(ef_adaptor_path);
        hnswdis::EfAdapter ef_adapter(
            hnsw, data, k, "cd", expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, sampling_size, min_queries_per_score);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "EF-estimation table computing time: " << duration << " ms" << std::endl;
        {
            hnswdis::MatrixXf _sq; hnswdis::MatrixXi _sgt;
            hnswdis::deserialize_samplings(samplings_path, _sq, _sgt);
            train_cv_buckets(ef_adapter, hnsw, data,
                k, "cd", alpha, gamma, statics_length,
                std::make_shared<hnswdis::MatrixXf>(_sq),
                std::make_shared<hnswdis::MatrixXi>(_sgt),
        ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);
        }
        ef_adapter.serialize(ef_adaptor_path);
    }
}

void process_offline_conf(const ExperimentConfig& conf, bool fast_rebuild)
{
    std::string dataset = conf.dataset;
    std::string metric = conf.metric;
    float alpha = conf.alpha;
    float expected_recall = conf.expected_recall;
    int ef_upper_bound = conf.ef_upper_bound;
    int sampling_size = conf.sampling_size;
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
    size_t statics_length = conf.statics_length;
    size_t k = conf.k;
    float gamma = conf.gamma;

    if (!fast_rebuild) {
        std::cout << "\nDataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Truncation ratio: " << alpha << std::endl;
    } else {
        std::cout << "Rebuilding table for " << dataset << "..." << std::endl;
    }

    if (dataset == "laion_text")
    {
        offline_laion_text2image();
        return;
    }

    std::string hdf5_path = (root / ("data/" + dataset + ".hdf5")).string();
    std::string index_path = (root / ("index/" + dataset + "-M16-efc-500-parallel.hnsw")).string();

    auto [hnsw, query, data, ground_truth, space] = load_index_and_data(hdf5_path, index_path, metric);

    std::string ef_adaptor_path = (root / ("estimation_table/" + dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table
    std::string samplings_path = (root / ("sampling/" + dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();           // path for sampling (queries and ground truth)

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    // 1. Sample data and compute ground truth
    start = std::chrono::high_resolution_clock::now();
    size_t num_hard_queries = 0;
    auto pair = hnswdis::compute_samplings(hnsw, data, metric, k, sampling_size, alpha, gamma, statics_length, 30000, &num_hard_queries);
    end = std::chrono::high_resolution_clock::now();
    if (!fast_rebuild) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Sampling computing time: " << duration << " ms" << std::endl;
    }
    hnswdis::serialize_samplings(samplings_path, pair.first, pair.second);
    {
        std::ofstream meta_out(samplings_path + ".meta");
        meta_out << num_hard_queries << "\n";
    }

    // 2. compute ef_adaptor
    start = std::chrono::high_resolution_clock::now();
    hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, sampling_size, min_queries_per_score);
    end = std::chrono::high_resolution_clock::now();
    if (!fast_rebuild) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "EF-estimation table computing time: " << duration << " ms" << std::endl;
    }

    hnswdis::MatrixXf _sq; hnswdis::MatrixXi _sgt;
    hnswdis::deserialize_samplings(samplings_path, _sq, _sgt);
    train_cv_buckets(ef_adapter, hnsw, data,
        k, metric, alpha, gamma, statics_length,
        std::make_shared<hnswdis::MatrixXf>(_sq),
        std::make_shared<hnswdis::MatrixXi>(_sgt),
        ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);

    ef_adapter.serialize(ef_adaptor_path);
    if (fast_rebuild) {
        std::cout << "Finished building table for " << dataset << std::endl;
    }
}

#include <future>

void offline_exp(bool fast_rebuild = false)
{
    std::cout << "Starting offline experiments...\n" << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Same threads logic as ablation studies

    for (const auto& conf : g_experiments) {
        process_offline_conf(conf, fast_rebuild);
    }
}

void compute_groundtruth_laion_text2image()
{
    std::string laion_text = (root / "data/laion_text_query_groundtruth.hdf5").string(); // query: text, data: text
    hnswdis::MatrixXf text_query_vectors;
    load_hdf5(laion_text, "test", text_query_vectors); // load the text query only
    std::cout << "text_query_vectors: " << text_query_vectors.rows() << ", " << text_query_vectors.cols() << std::endl;

    std::string laion_image = (root / "data/laion_image.hdf5").string(); // query: image, data: image
    hnswdis::MatrixXf image_data_vectors;
    load_hdf5(laion_image, "train", image_data_vectors); // load the image data only
    std::cout << "image_data_vectors: " << image_data_vectors.rows() << ", " << image_data_vectors.cols() << std::endl;

    // normalize the vectors: consine distance
    normalize_matrix(text_query_vectors);
    normalize_matrix(image_data_vectors);

    std::cout << "Computing ground truth for text2image retrieval..." << std::endl;
    // computing ground truth for text2image retrieval
    auto ground_truth = hnswdis::compute_ground_truth(text_query_vectors, image_data_vectors, "cd", 1000);

    std::string gt_path = (root / "data/laion_text_query_groundtruth.hdf5").string();

    // save the ground truth to hdf5 file
    save_hdf5(gt_path, text_query_vectors, ground_truth);
}

void sensitivity_analysis()
{
    std::cout << "Starting sensitivity analysis...\n\n\n"
              << std::endl;
    std::vector<std::string> datasets = {
        "cohere",
        "deep-image-96-angular1000"};

    for (const std::string &dataset : datasets)
    {
        auto conf = get_config(dataset);
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        std::string metric = "cd";
        float alpha = 0.25f;
        float gamma = 12.0f;
        std::vector<int> list_k = {1000, 100, 50}; // in descending order so that we can reuse samplings and estimators more effectively
        std::vector<float> expected_recalls = {0.95, 0.97, 0.99};
        int ef_upper_bound = 5000;

        // samplings and estimators are k agnostic and can be reused across different k values; they are also expected recall agnostic and can be reused across different expected recall values
        std::string samplings_path = (root / ("sampling/" + dataset + "-samplings-" + "-k1000-ef.bin")).string(); // path for sampling (queries and ground truth)

        std::string hdf5_path = (root / ("data/" + dataset + ".hdf5")).string();
        std::string index_path = (root / ("index/" + dataset + "-M16-efc-500-parallel.hnsw")).string();

        if (dataset == "deep-image-96-angular1000")
        {
            index_path = (root / "index/deep-image-96-angular-M16-efc-500-parallel.hnsw").string();     // reuse the index of deep-image-96-angular
            // sampling will recomputed on the fly for deep image because it requires a larger k value, i.e., 1000, and the existing sampling is for k=100
            // in the case of vese versa, i.e., reusing a sampling with larger k for a smaller k, it is safe to do so
        }

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;

        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        int repeat = 1;

        std::cout << "Dataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Truncation ratio: " << alpha << std::endl;

        for (const auto k : list_k)
        {

            for (const auto expected_recall : expected_recalls)
            {
                std::cout
                    << "k: " << k << std::endl
                    << "Expected recall: " << expected_recall << std::endl;

                start = std::chrono::high_resolution_clock::now();
                hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, conf.sampling_size);
                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "EF-estimation table computing time: " << duration << " ms" << std::endl;
                auto ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);
                hnswdis::Sketch sketch = make_sketch(*ef_adapter_ptr, expected_recall);
                std::cout << "print sketch" << std::endl;
                sketch.print();
                const float wae = ef_adapter_ptr->get_wae();
                std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;

                hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);


                hnsw->setEf(wae);
                adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
            }

            // baseline search with different ef values
            baseline_search(dataset, repeat, *hnsw, *query, *ground_truth, k, ef_upper_bound);
        }
    }
}

void insert_exp_setup(
    const std::string &dataset,
    const std::string &metric,
    const hnswdis::MatrixXf &before_updates_data,
    const size_t &total_num_points,
    const size_t k,
    const std::string &batch_type)
{

    auto before_data_ptr = std::make_shared<hnswdis::MatrixXf>(before_updates_data);
    size_t num_points_before_updates = before_updates_data.rows();

    // start building the index
    size_t dim = before_updates_data.cols();
    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, dim);
    // initialize the index, M=16, ef_construction=500, max_elements=full_data.rows()
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), total_num_points, 16, 500);

    // add the points in before_updates_data to the index
    auto start = std::chrono::high_resolution_clock::now();
    hnswdis::ParallelFor(0, num_points_before_updates, std::max(1u, std::thread::hardware_concurrency() / 4), [&](size_t row_id, size_t threadId)
                         { alg_hnsw->addPoint((void *)(before_updates_data.row(row_id).data()), row_id); });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index built in " << duration.count() << " ms" << std::endl;
    std::cout << "Index size: " << alg_hnsw->cur_element_count << std::endl;
    std::cout << "Index capacity: " << alg_hnsw->getMaxElements() << std::endl;

    // save the index to disk
    std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    alg_hnsw->saveIndex(index_path);

    // compute estimator

    start = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // save the estimator to disk

    // compute samplings
    std::string samplings_path = (root / "incremental_update" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
    start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors = hnswdis::sample_data(before_updates_data, 200);
    auto [sample_ground_truth, sample_ground_truth_dist] = hnswdis::compute_ground_truth_batch_parallel4_with_dist(*sample_query_vectors, before_updates_data, metric, 2 * k);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sampling computing time: " << duration.count() << " ms" << std::endl;
    // save the samplings to disk
    hnswdis::serialize_samplings(samplings_path, *sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);

    // compute ef_adaptor
    std::string ef_adaptor_path = (root / "incremental_update" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string();
    float expected_recall = 0.95;
    auto conf = get_config(dataset);
    int sampling_size = conf.sampling_size;
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
    size_t statics_length = conf.statics_length;
    float gamma = conf.gamma;
    float alpha = conf.alpha;
    int ef_upper_bound = conf.ef_upper_bound;


    start = std::chrono::high_resolution_clock::now();
    hnswdis::EfAdapter ef_adapter(alg_hnsw, before_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, sampling_size, min_queries_per_score);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "EF-estimation table computing time: " << duration.count() << " ms" << std::endl;
    train_cv_buckets(ef_adapter, alg_hnsw, before_data_ptr,
        k, metric, alpha, gamma, statics_length,
        sample_query_vectors, std::make_shared<hnswdis::MatrixXi>(sample_ground_truth),
        ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);
    ef_adapter.serialize(ef_adaptor_path);
}

void insert_exp_index_update(
    const std::string &dataset,
    const std::string &metric,
    const hnswdis::MatrixXf &full_data,
    const hnswdis::MatrixXf &before_updates_data,
    const size_t &num_updates,
    const std::string &batch_type)
{
    std::cout << "\n\nUpdate the index with the new data\n\n"
              << std::endl;
    std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    std::cout << "Index path: " << index_path << std::endl;
    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, before_updates_data.cols());
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    std::cout << "Index loaded" << std::endl;

    // Update the index with the new data
    std::cout << "Update the index with the new data" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // add the points in before_updates_data to the index
    hnswdis::ParallelFor(full_data.rows() - num_updates, full_data.rows(), std::max(1u, std::thread::hardware_concurrency() / 4), [&](size_t row_id, size_t threadId)
                         { alg_hnsw->addPoint((void *)(full_data.row(row_id).data()), row_id); });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index updated in " << duration.count() << " ms" << std::endl;
    std::cout << "Index size: " << alg_hnsw->cur_element_count << std::endl;
    std::cout << "Index capacity: " << alg_hnsw->getMaxElements() << std::endl;

    // save the updated index to disk
    std::string updated_index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel-updated.hnsw")).string();
    alg_hnsw->saveIndex(updated_index_path);
}

void insert_exp_adaef_update(
    const std::string &dataset,
    const std::string &metric,
    const hnswdis::MatrixXf &full_data,
    const hnswdis::MatrixXf &before_updates_data,
    const hnswdis::MatrixXf &updates_data,
    const size_t k,
    const float expected_recall,
    const float alpha,
    const float gamma,
    const size_t statics_length,
    const int ef_upper_bound,
    const int before_updates,
    const std::string &batch_type)
{
    std::cout << "\n\nUpdate the ef_adaptor with the new data\n\n"
              << std::endl;

    // load the updated index
    std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel-updated.hnsw")).string();
    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, before_updates_data.cols());
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    std::cout << "Updated index loaded" << std::endl;

    for (size_t i = 0; i < 3; i++)
    {
        // load stale estimator

        // load stale samplings
        std::string samplings_path = (root / "incremental_update" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        std::cout << "Samplings loaded" << std::endl;

        // update estimator
            auto conf = get_config(dataset);
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // update samplings
        start = std::chrono::high_resolution_clock::now();
        hnswdis::update_ground_truth_with_new_data(
            sample_query_vectors,
            sample_ground_truth,
            sample_ground_truth_dist,
            updates_data,
            before_updates);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Samplings updated in " << duration.count() << " ms" << std::endl;
        std::string updated_samplings_path = (root / "incremental_update" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
        hnswdis::serialize_samplings(updated_samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);

        // compute adaptor
        std::shared_ptr<hnswdis::MatrixXf> full_data_ptr = std::make_shared<hnswdis::MatrixXf>(full_data);
        std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>(sample_query_vectors);
        std::shared_ptr<hnswdis::MatrixXi> sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(sample_ground_truth);

        start = std::chrono::high_resolution_clock::now();
        hnswdis::EfAdapter ef_adapter(
            alg_hnsw, full_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors_ptr, sample_ground_truth_ptr, ef_upper_bound, min_queries_per_score);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "EF-estimation table computing time: " << duration.count() << " ms" << std::endl;
        std::string updated_ef_adaptor_path = (root / "incremental_update" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
        train_cv_buckets(ef_adapter, alg_hnsw, full_data_ptr,
            k, metric, alpha, gamma, statics_length,
            sample_query_vectors_ptr, sample_ground_truth_ptr,
            ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);
        ef_adapter.serialize(updated_ef_adaptor_path);
    }
}

void insert_exp(bool setup = false)
{
    // large batch incremental insert experiment with 10% new data
    std::cout << "Starting incremental insert experiments...\n\n"
              << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 of all available threads
    std::vector<std::tuple<std::string, size_t, std::string, size_t>> dataset_updates = {
        {"cohere", 1000, "10percent", 1838056},
        {"deep-image-96-angular", 100, "10percent", 999000},
        {"cohere", 1000, "50percent", 9190280},
        {"deep-image-96-angular", 100, "50percent", 4995000},
    };

    const float expected_recall = 0.95;

    for (const auto [dataset, k, batch_type, num_updates] : dataset_updates)
    {

        std::cout << "\n\nDataset: " << dataset << ", k: " << k << ", batch type: " << batch_type << ", num_updates: " << num_updates << "\n\n"
                  << std::endl;

        std::string metric = "cd";
        auto conf = get_config(dataset);
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;
        float alpha = conf.alpha;
        int ef_upper_bound = conf.ef_upper_bound;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        hnswdis::MatrixXf full_data;
        hnswdis::MatrixXf query_vectors;
        hnswdis::MatrixXi ground_truth;

        load_hdf5(hdf5_path, query_vectors, full_data, ground_truth);

        // normalize the data
        normalize_matrix(full_data);
        normalize_matrix(query_vectors);

        // data before inserts
        std::cout << "Dataset: " << dataset << std::endl;
        std::cout << "full_data: " << full_data.rows() << ", " << full_data.cols() << std::endl;
        int before_updates = full_data.rows() - num_updates;

        // get data before updates
        hnswdis::MatrixXf before_updates_data = full_data.topRows(before_updates);
        hnswdis::MatrixXf updates_data = full_data.bottomRows(num_updates);

        // setup for the experiment
        if (setup)
        {
            insert_exp_setup(dataset, metric, before_updates_data, full_data.rows(), k, batch_type);
            insert_exp_index_update(dataset, metric, full_data, before_updates_data, num_updates, batch_type);
            insert_exp_adaef_update(dataset, metric, full_data, before_updates_data, updates_data, k, expected_recall, alpha, gamma, statics_length, ef_upper_bound, before_updates, batch_type);
        }

        int repeat = 6;

        // load updated index
        std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel-updated.hnsw")).string();
        std::cout << "Index path: " << index_path << std::endl;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, before_updates_data.cols());
        std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
        std::cout << "Updated index loaded" << std::endl;

        // stale performance: use stale (unupdated) estimator, samplings, and adaptor to do search
        {
            std::cout << "\n\nStale performance: use stale (unupdated) estimator, samplings, and adaptor to do search\n\n"
                      << std::endl;
            // load estimator

            // load samplings
            std::string samplings_path = (root / "incremental_update" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
            hnswdis::MatrixXf sample_query_vectors;
            hnswdis::MatrixXi sample_ground_truth;
            hnswdis::MatrixXf sample_ground_truth_dist;
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
            std::cout << "Samplings loaded" << std::endl;

            // load ef_adaptor
            std::string ef_adaptor_path = (root / "incremental_update" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string();
            std::shared_ptr<hnswdis::EfAdapter> ef_adapter_ptr;
            hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
            ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);

            hnswdis::Sketch sketch = make_sketch(*ef_adapter_ptr, expected_recall);
            const float wae = ef_adapter_ptr->get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
                        alg_hnsw->setEf(wae);
            adaptive_search(dataset, repeat, *alg_hnsw, query_vectors, full_data, ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }

        // updated performance: use updated estimator, samplings, and adaptor to do search
        {
            std::cout << "\n\nUpdated performance: use updated estimator, samplings, and adaptor to do search\n\n"
                      << std::endl;

            // load updated estimator
            std::cout << "Updated estimator loaded" << std::endl;

            // load the updated adaptor
            std::string updated_ef_adaptor_path = (root / "incremental_update" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
            hnswdis::EfAdapter ef_adapter(updated_ef_adaptor_path);
            std::cout << "Updated ef_adaptor loaded" << std::endl;

            // perform the search
            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;

            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
                        alg_hnsw->setEf(wae);
            // shiro-ef search with updated estimator, samplings, and adaptor
            adaptive_search(dataset, repeat, *alg_hnsw, query_vectors, full_data, ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void delete_exp_setup(
    const std::string &dataset,
    const std::string &metric,
    hnswdis::MatrixXf &after_updates_data,
    const hnswdis::MatrixXf &full_data, // used only for building full ground truth for samplings
    hnswdis::MatrixXf &query_vectors,
    const size_t k,
    const std::string &batch_type)
{
    // compute the groundtruth of the query vectors over the after_updates_data (data after deletions)
    {
        hnswdis::MatrixXi after_updates_ground_truth = hnswdis::compute_ground_truth(query_vectors, after_updates_data, metric, k);
        std::cout << "After updates ground truth computed" << std::endl;
        // store the ground truth for later use
        std::string ground_truth_path = (root / "incremental_deletion" / batch_type / (dataset + "-after_updates-ground-truth-" + "-k-" + std::to_string(k) + ".hdf5")).string();
        save_hdf5(ground_truth_path, query_vectors, after_updates_ground_truth); // this is used for query results evaluation
    }

    // compute the full groundtruth of sampling query vectors over the full data (before deletions), used for sampling update
    {
        // load stale samplings
        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();

        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth_k_only;
        hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth_k_only);
        std::cout << "Samplings loaded" << std::endl;
        auto sample_ground_truth_full_dist = hnswdis::build_full_gt_structure(sample_query_vectors, full_data);
        std::cout << "Full ground truth for samplings computed" << std::endl;
        // store the full ground truth for later use
        std::string full_ground_truth_path = (root / "incremental_deletion" / batch_type / (dataset + "-full-ground-truth-samplings-" + "-k-" + std::to_string(k) + ".bin")).string();
        hnswdis::save_query_heaps(sample_ground_truth_full_dist, full_ground_truth_path);
    }

    // recompute from scratch
    {
        // load the updated index: reuse the index before insertion in the insert experiments, which is equivalent to the index after deletions
        std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, after_updates_data.cols());
        std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
        std::cout << "Updated index loaded" << std::endl;

        // compute estimator
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // save the estimator to disk

        // compute samplings
        std::string samplings_path = (root / "incremental_deletion" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef-recomp.bin")).string();
        start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors = hnswdis::sample_data(after_updates_data, 200);
        auto [sample_ground_truth, sample_ground_truth_dist] = hnswdis::compute_ground_truth_batch_parallel4_with_dist(*sample_query_vectors, after_updates_data, metric, k);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Sampling computing time: " << duration.count() << " ms" << std::endl;
        // save the samplings to disk
        hnswdis::serialize_samplings(samplings_path, *sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);

        // compute ef_adaptor
        std::string ef_adaptor_path = (root / "incremental_deletion" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-recomp.bin")).string();
        float expected_recall = 0.95;
    auto conf = get_config(dataset);
    int sampling_size = conf.sampling_size;
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
    size_t statics_length = conf.statics_length;
    float gamma = conf.gamma;
    float alpha = conf.alpha;
    int ef_upper_bound = conf.ef_upper_bound;

                                        int k = 1000;

        std::shared_ptr<hnswdis::MatrixXf> after_updates_data_ptr = std::make_shared<hnswdis::MatrixXf>(after_updates_data);
        std::shared_ptr<hnswdis::MatrixXi> sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(sample_ground_truth);
        start = std::chrono::high_resolution_clock::now();
        hnswdis::EfAdapter ef_adapter(alg_hnsw, after_updates_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors, sample_ground_truth_ptr, ef_upper_bound, min_queries_per_score);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "EF-estimation table computing time: " << duration.count() << " ms" << std::endl;
        train_cv_buckets(ef_adapter, alg_hnsw, after_updates_data_ptr,
            k, metric, alpha, gamma, statics_length,
            sample_query_vectors, sample_ground_truth_ptr,
            ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);
        ef_adapter.serialize(ef_adaptor_path);
    }
}

void delete_exp_adaef_update(
    const std::string &dataset,
    const std::string &metric,
    const int full_data_rows,
    const hnswdis::MatrixXf &after_updates_data,
    const hnswdis::MatrixXf &updates_data,
    const hnswdis::MatrixXf &query_vectors,
    const size_t k,
    const float expected_recall,
    const float alpha,
    const float gamma,
    const size_t statics_length,
    const int ef_upper_bound,
    const std::string &batch_type)
{
    // load the updated index: reuse the index before insertion in the insert experiments, which is equivalent to the index after deletions
    std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel.hnsw")).string();
    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, after_updates_data.cols());
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    std::cout << "Updated index loaded" << std::endl;

    // load stale estimator: in this case, the stale estimator is full estimater before deletions

    // load stale sampling: in this case, the stale sampling is full samplings ground truth before deletions, computed by deletion_exp_setup()
    std::string full_ground_truth_path = (root / "incremental_deletion" / batch_type / (dataset + "-full-ground-truth-samplings-" + "-k-" + std::to_string(k) + ".bin")).string();
    auto sample_ground_truth_full_dist = hnswdis::load_query_heaps(full_ground_truth_path);
    std::cout << "Full ground truth for samplings loaded" << std::endl;

    // update estimator
        auto conf = get_config(dataset);
    int n_cv_tables = conf.n_cv_tables;
    int min_queries_per_score = conf.min_queries_per_score;
auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // update samplings
    hnswdis::MatrixXi sample_ground_truth;      // create ground truth vectors
    hnswdis::MatrixXf sample_ground_truth_dist; // create ground truth vectors
    start = std::chrono::high_resolution_clock::now();
    std::unordered_set<int> deleted_ids;
    for (size_t i = 0; i < updates_data.rows(); i++)
    {
        deleted_ids.insert(full_data_rows - updates_data.rows() + i);
    }
    hnswdis::gt_delete_points(sample_ground_truth_full_dist, deleted_ids);
    hnswdis::export_topk(sample_ground_truth_full_dist, k, sample_ground_truth, sample_ground_truth_dist);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Samplings updated in " << duration.count() << " ms" << std::endl;
    // before saving the update samplings, first load sampling query vectors
    hnswdis::MatrixXf sample_query_vectors;
    hnswdis::MatrixXi sample_ground_truth_k_only; // dummy
    hnswdis::deserialize_samplings(               // the original sampling query vectors are used, which is also used in deletion_exp_setup()
        (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string(),
        sample_query_vectors,
        sample_ground_truth_k_only);
    std::string updated_samplings_path = (root / "incremental_deletion" / batch_type / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
    hnswdis::serialize_samplings(updated_samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);

    // compute adaptor
    std::shared_ptr<hnswdis::MatrixXf> after_updates_data_ptr = std::make_shared<hnswdis::MatrixXf>(after_updates_data);
    std::shared_ptr<hnswdis::MatrixXf> sample_query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>(sample_query_vectors);
    std::shared_ptr<hnswdis::MatrixXi> sample_ground_truth_ptr = std::make_shared<hnswdis::MatrixXi>(sample_ground_truth);
    start = std::chrono::high_resolution_clock::now();
    hnswdis::EfAdapter ef_adapter(
        alg_hnsw, after_updates_data_ptr, k, metric, expected_recall, alpha, gamma, statics_length, sample_query_vectors_ptr, sample_ground_truth_ptr, ef_upper_bound, min_queries_per_score);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "EF-estimation table computing time: " << duration.count() << " ms" << std::endl;
    std::string updated_ef_adaptor_path = (root / "incremental_deletion" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
    train_cv_buckets(ef_adapter, alg_hnsw, after_updates_data_ptr,
        k, metric, alpha, gamma, statics_length,
        sample_query_vectors_ptr, sample_ground_truth_ptr,
        ef_upper_bound, n_cv_tables, min_queries_per_score, updated_samplings_path);
    ef_adapter.serialize(updated_ef_adaptor_path);
}

void delete_exp(bool setup = false)
{
    std::cout << "Starting incremental delete experiments...\n\n"
              << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 of all available threads
    std::vector<std::tuple<std::string, size_t, std::string, size_t>> dataset_updates = {
        {"cohere", 1000, "10percent", 1838056},
        {"deep-image-96-angular", 100, "10percent", 999000},
        {"cohere", 1000, "50percent", 9190280},
        {"deep-image-96-angular", 100, "50percent", 4995000},
    };

    const float expected_recall = 0.95;

    for (const auto [dataset, k, batch_type, num_updates] : dataset_updates)
    {

        std::cout << "\n\nDataset: " << dataset << ", k: " << k << ", batch type: " << batch_type << ", num_updates: " << num_updates << "\n\n"
                  << std::endl;

        std::string metric = "cd";
        auto conf = get_config(dataset);
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;
        float alpha = conf.alpha;
        int ef_upper_bound = conf.ef_upper_bound;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();

        hnswdis::MatrixXf full_data;
        hnswdis::MatrixXf query_vectors;
        hnswdis::MatrixXi full_ground_truth; // the full ground truth before deletions

        load_hdf5(hdf5_path, query_vectors, full_data, full_ground_truth);

        // normalize the data
        normalize_matrix(full_data);
        normalize_matrix(query_vectors);

        // data before inserts
        std::cout << "Dataset: " << dataset << std::endl;
        std::cout << "full_data: " << full_data.rows() << ", " << full_data.cols() << std::endl;

        // get data before updates
        hnswdis::MatrixXf updates_data = full_data.bottomRows(num_updates);
        hnswdis::MatrixXf after_updates_data = full_data.topRows(full_data.rows() - num_updates);

        // setup for the experiment
        if (setup)
        {
            delete_exp_setup(dataset, metric, after_updates_data, full_data, query_vectors, k, batch_type);
            delete_exp_adaef_update(dataset, metric, full_data.rows(), after_updates_data, updates_data, query_vectors, k, expected_recall, alpha, gamma, statics_length, ef_upper_bound, batch_type);
        }

        // get the ground truth after deletions
        hnswdis::MatrixXf dumm_query; // this is the same as query_vectors
        hnswdis::MatrixXi ground_truth;
        load_hdf5((root / "incremental_deletion" / batch_type / (dataset + "-after_updates-ground-truth-" + "-k-" + std::to_string(k) + ".hdf5")).string(),
                  "test", dumm_query,
                  "neighbors", ground_truth);

        // load updated index, which is the same as the index before insertions in the insertion experiments
        std::string index_path = (root / "incremental_update" / batch_type / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, after_updates_data.cols());
        std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
        std::cout << "Updated index loaded" << std::endl;

        std::cout << "Starting searches...\n\n"
                  << std::endl;

        int repeat = 6;
        // stale performance: use stale (unupdated) estimator, samplings, and adaptor to do search, in this case, the stale ones are the full ones before deletions
        {
            std::cout << "\n\nStale performance: use stale (unupdated) estimator, samplings, and adaptor to do search\n\n"
                      << std::endl;

            std::string ef_adaptor_path = (root / "estimation_table" / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table
            std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();           // path for sampling (queries and ground truth)

            // load estimator

            // load samplings
            hnswdis::MatrixXf sample_query_vectors;
            hnswdis::MatrixXi sample_ground_truth;
            hnswdis::MatrixXf sample_ground_truth_dist;
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
            std::cout << "Samplings loaded" << std::endl;

            // load ef_adaptor
            std::shared_ptr<hnswdis::EfAdapter> ef_adapter_ptr;
            hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
            ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);

            hnswdis::Sketch sketch = make_sketch(*ef_adapter_ptr, expected_recall);
            const float wae = ef_adapter_ptr->get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
                        alg_hnsw->setEf(wae);
            adaptive_search(dataset, repeat, *alg_hnsw, query_vectors, after_updates_data, ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }

        // updated performance: use updated estimator, samplings, and adaptor to do search
        {
            std::cout << "\n\nUpdated performance: use updated estimator, samplings, and adaptor to do search\n\n"
                      << std::endl;

            // load updated estimator
            std::cout << "Updated estimator loaded" << std::endl;

            // load the updated adaptor
            std::string updated_ef_adaptor_path = (root / "incremental_deletion" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-updated.bin")).string();
            hnswdis::EfAdapter ef_adapter(updated_ef_adaptor_path);
            std::cout << "Updated ef_adaptor loaded" << std::endl;

            // perform the search
            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;

            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
                        alg_hnsw->setEf(wae);
            // shiro-ef search with updated estimator, samplings, and adaptor
            adaptive_search(dataset, repeat, *alg_hnsw, query_vectors, after_updates_data, ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }

        // experiments with recomputed adaef
        {
            std::cout << "\n\nPerformance with recomputed adaef: use recomputed estimator, samplings, and adaptor to do search\n\n"
                      << std::endl;

            // load recomputed estimator
            std::cout << "Recomputed estimator loaded" << std::endl;

            // load the recomputed adaptor
            std::string recomp_ef_adaptor_path = (root / "incremental_deletion" / batch_type / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef-recomp.bin")).string();
            hnswdis::EfAdapter ef_adapter(recomp_ef_adaptor_path);
            std::cout << "Recomputed ef_adaptor loaded" << std::endl;

            // perform the search
            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;

            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
                        alg_hnsw->setEf(wae);
            // shiro-ef search with recomputed estimator, samplings, and adaptor
            adaptive_search(dataset, repeat, *alg_hnsw, query_vectors, after_updates_data, ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void ablation_study_visited_list_size()
{
    std::cout << "Starting ablation study tests: visited list size...\n\n"
              << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 available threads for eigen parallelization in shiro-ef offline computation

    std::vector<int> visited_list_sizes = {
        1 + 32,                          // 1-hop neighbors on the base layer: M = 16
        1 + 32 + 31 * 32,                // 2-hop neighbors on the base layer: M = 16
        1 + 32 + 31 * 32 + 31 * 32 * 32, // 3-hop neighbors on the base layer: M = 16
    };

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;
        int repeat = 3;

        std::cout << "Dataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Truncation ratio: " << alpha << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);
        space = std::get<4>(tuple);

        // the followings are for adaptive ef experiments
        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for sampling (queries and ground truth)

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        // 1. load estimator
        hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

        // 2. load samplings
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        std::cout << "Samplings loaded" << std::endl;

        for (const auto statics_length : visited_list_sizes)
        {
            std::cout << "\n\nVisited list size: " << statics_length << "\n\n"
                      << std::endl;

            std::string ef_adaptor_path = (root / "ablation_distance_size" / (dataset + "-D-" + std::to_string(statics_length) + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table

            auto start = std::chrono::high_resolution_clock::now();
            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, conf.sampling_size);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "EF-estimation table computing time: " << duration << " ms" << std::endl;
            train_cv_buckets(ef_adapter, hnsw, data,
                k, metric, alpha, gamma, statics_length,
                std::make_shared<hnswdis::MatrixXf>(sample_query_vectors), std::make_shared<hnswdis::MatrixXi>(sample_ground_truth),
                ef_upper_bound, conf.n_cv_tables, min_queries_per_score, samplings_path);
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnsw->setEf(wae);
            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void ablation_study_sampling_size()
{
    std::cout << "Starting ablation study tests: sampling size...\n\n"
              << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 available threads for eigen parallelization in shiro-ef offline computation

    std::vector<int> sampling_size = {
        // from larger to smaller, reuse samplings for experiments
        5000,
        3000,
        2000,
        1000,
    };

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;
        int repeat = conf.repeat;

        std::cout << "Dataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Truncation ratio: " << alpha << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);
        space = std::get<4>(tuple);

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 1. load estimator
        hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

        for (const auto samplings : sampling_size)
        {
            std::cout << "\n\nSampling size: " << samplings << "\n\n"
                      << std::endl;

            std::string samplings_path = (root / "ablation_sampling_size" / (dataset + "-samplings-" + std::to_string(samplings) + "-k" + std::to_string(k) + "-ef.bin")).string();                   // path for sampling (queries and ground truth)
            std::string ef_adaptor_path = (root / "ablation_sampling_size" / (dataset + "-samplings-" + std::to_string(samplings) + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table

            start = std::chrono::high_resolution_clock::now();
            auto pair = hnswdis::compute_samplings(hnsw, data, metric, k, samplings, alpha, gamma, statics_length);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Sampling computing time: " << duration << " ms" << std::endl;
            hnswdis::serialize_samplings(samplings_path, pair.first, pair.second);

            start = std::chrono::high_resolution_clock::now();
            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, samplings);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "EF-estimation table computing time: " << duration << " ms" << std::endl;
            train_cv_buckets(ef_adapter, hnsw, data,
                k, metric, alpha, gamma, statics_length,
                std::make_shared<hnswdis::MatrixXf>(pair.first), std::make_shared<hnswdis::MatrixXi>(pair.second),
                ef_upper_bound, conf.n_cv_tables, min_queries_per_score, samplings_path);
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnsw->setEf(wae);
            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void ablation_study_weighted_decay_function()
{
    std::cout << "Starting ablation study tests: weighted decay function (gamma)...\n\n" << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    std::vector<float> gammas = {8.0f, 12.0f, 16.0f, 24.0f, 32.0f};

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        int repeat = conf.repeat;

        std::cout << "\n\nDataset: " << dataset << "\n" << "Metric: " << metric << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);
        space = std::get<4>(tuple);

        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        try {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth);
        }


        for (const auto gamma : gammas)
        {
            std::cout << "\n--- Gamma: " << gamma << " ---\n" << std::endl;

            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

            std::string ef_adaptor_path = (root / "ablation_gamma" / (dataset + "-gamma-" + std::to_string(gamma) + "-ef.bin")).string();

            auto start = std::chrono::high_resolution_clock::now();
            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, conf.sampling_size);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            train_cv_buckets(ef_adapter, hnsw, data, k, metric, alpha, gamma, statics_length,
                             std::make_shared<hnswdis::MatrixXf>(sample_query_vectors),
                             std::make_shared<hnswdis::MatrixXi>(sample_ground_truth),
                             ef_upper_bound, conf.n_cv_tables, min_queries_per_score, samplings_path);

            std::filesystem::create_directories(root / "ablation_gamma");
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnsw->setEf(wae);

            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void ablation_study_truncation_ratio()
{
    std::cout << "Starting ablation study tests: truncation ratio (alpha)...\n\n" << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    std::vector<float> alphas = {0.25f, 0.5f, 0.75f};

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float gamma = conf.gamma;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        int repeat = 3;

        std::cout << "\n\nDataset: " << dataset << "\n" << "Metric: " << metric << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);
        space = std::get<4>(tuple);

        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        try {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth);
        }


        for (const auto alpha : alphas)
        {
            std::cout << "\n--- Alpha: " << alpha << " ---\n" << std::endl;

            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

            std::string ef_adaptor_path = (root / "ablation_alpha" / (dataset + "-alpha-" + std::to_string(alpha) + "-ef.bin")).string();

            auto start = std::chrono::high_resolution_clock::now();
            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, conf.sampling_size);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            train_cv_buckets(ef_adapter, hnsw, data, k, metric, alpha, gamma, statics_length,
                             std::make_shared<hnswdis::MatrixXf>(sample_query_vectors),
                             std::make_shared<hnswdis::MatrixXi>(sample_ground_truth),
                             ef_upper_bound, conf.n_cv_tables, min_queries_per_score, samplings_path);

            std::filesystem::create_directories(root / "ablation_alpha");
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            const float wae = ef_adapter.get_wae();
            std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
            hnsw->setEf(wae);

            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}


void ablation_study_n_cv_tables()
{
    std::cout << "Starting ablation study tests: n_cv_tables...\n\n" << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    std::vector<int> tables_list = {0, 5, 10, 15, 20};

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        float gamma = conf.gamma;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        int repeat = 3;

        std::cout << "\n\nDataset: " << dataset << "\n" << "Metric: " << metric << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);

        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        try {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth);
        }

        for (const auto n_cv_tables : tables_list)
        {
            std::cout << "\n--- n_cv_tables: " << n_cv_tables << " ---\n" << std::endl;
            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
            std::string ef_adaptor_path = (root / "ablation_n_cv" / (dataset + "-ncv-" + std::to_string(n_cv_tables) + "-ef.bin")).string();

            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, sampling_size, min_queries_per_score);
            train_cv_buckets(ef_adapter, hnsw, data, k, metric, alpha, gamma, statics_length, std::make_shared<hnswdis::MatrixXf>(sample_query_vectors), std::make_shared<hnswdis::MatrixXi>(sample_ground_truth), ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);

            std::filesystem::create_directories(root / "ablation_n_cv");
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            hnsw->setEf(ef_adapter.get_wae());

            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void ablation_study_min_queries_per_score()
{
    std::cout << "Starting ablation study tests: min_queries_per_score...\n\n" << std::endl;
    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4));

    std::vector<int> min_q_list = {0, 1, 3, 5, 10};

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        float gamma = conf.gamma;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        size_t statics_length = conf.statics_length;
        int repeat = 3;

        std::cout << "\n\nDataset: " << dataset << "\n" << "Metric: " << metric << std::endl;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
        std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
        auto tuple = load_index_and_data(hdf5_path, index_path, metric);
        hnsw = std::get<0>(tuple);
        query = std::get<1>(tuple);
        data = std::get<2>(tuple);
        ground_truth = std::get<3>(tuple);

        std::string samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXf sample_query_vectors;
        hnswdis::MatrixXi sample_ground_truth;
        hnswdis::MatrixXf sample_ground_truth_dist;
        try {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth, sample_ground_truth_dist);
        } catch (...) {
            hnswdis::deserialize_samplings(samplings_path, sample_query_vectors, sample_ground_truth);
        }

        for (const auto min_queries_per_score : min_q_list)
        {
            std::cout << "\n--- min_queries_per_score: " << min_queries_per_score << " ---\n" << std::endl;
            hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);
            std::string ef_adaptor_path = (root / "ablation_min_q" / (dataset + "-minq-" + std::to_string(min_queries_per_score) + "-ef.bin")).string();

            hnswdis::EfAdapter ef_adapter(hnsw, data, k, metric, expected_recall, alpha, gamma, statics_length, samplings_path, ef_upper_bound, sampling_size, min_queries_per_score);
            train_cv_buckets(ef_adapter, hnsw, data, k, metric, alpha, gamma, statics_length, std::make_shared<hnswdis::MatrixXf>(sample_query_vectors), std::make_shared<hnswdis::MatrixXi>(sample_ground_truth), ef_upper_bound, n_cv_tables, min_queries_per_score, samplings_path);

            std::filesystem::create_directories(root / "ablation_min_q");
            ef_adapter.serialize(ef_adaptor_path);

            hnswdis::Sketch sketch = make_sketch(ef_adapter, expected_recall);
            hnsw->setEf(ef_adapter.get_wae());

            adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }
    }
}

void per_query_result_exp()
{
    std::cout << "Starting per-query result experiments...\n\n"
              << std::endl;

    Eigen::setNbThreads(std::max(1u, std::thread::hardware_concurrency() / 4)); // Limit to 1/4 available threads for eigen parallelization in shiro-ef offline computation

    for (const auto& conf : g_experiments)
    {
        std::string dataset = conf.dataset;
        std::string metric = conf.metric;
        float alpha = conf.alpha;
        size_t k = conf.k;
        float expected_recall = conf.expected_recall;
        int ef_upper_bound = conf.ef_upper_bound;
        int sampling_size = conf.sampling_size;
        int n_cv_tables = conf.n_cv_tables;
        int min_queries_per_score = conf.min_queries_per_score;
        size_t statics_length = conf.statics_length;
        float gamma = conf.gamma;

        std::cout << "Dataset: " << dataset << std::endl
                  << "Metric: " << metric << std::endl
                  << "Quantile step: " << alpha << std::endl;

        int repeat = 3;

        std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
        std::shared_ptr<hnswdis::MatrixXf> query;
        std::shared_ptr<hnswdis::MatrixXf> data;
        std::shared_ptr<hnswdis::MatrixXi> ground_truth;
        std::shared_ptr<hnswlib::SpaceInterface<float>> space;

        if (dataset == "laion_text")
        {
            setup_laion_text2image(hnsw, query, data, ground_truth, space);
        }
        else
        {
            std::string hdf5_path = (root / "data" / (dataset + ".hdf5")).string();
            std::string index_path = (root / "index" / (dataset + "-M16-efc-500-parallel.hnsw")).string();
            auto tuple = load_index_and_data(hdf5_path, index_path, metric);
            hnsw = std::get<0>(tuple);
            query = std::get<1>(tuple);
            data = std::get<2>(tuple);
            ground_truth = std::get<3>(tuple);
            space = std::get<4>(tuple);
        }

        // the followings are for adaptive ef experiments
        std::string ef_adaptor_path = (root / "estimation_table_o3" / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string(); // path for estimation table
        // std::string samplings_path = (root / "sampling_o3" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();           // path for sampling (queries and ground truth)

        if (dataset == "laion_text")
        {
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        // 1. load estimator
        hnswdis::ApproximatedScoreCalculator score_cal(alpha, gamma);

        // 2. load ef_adaptor
        std::shared_ptr<hnswdis::EfAdapter> ef_adapter_ptr;
        hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
        ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);

        // 3. create sketch
        hnswdis::Sketch sketch = make_sketch(*ef_adapter_ptr, expected_recall);
        const float wae = ef_adapter_ptr->get_wae();
        std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
                hnsw->setEf(wae);
        for (size_t i = 0; i < repeat; i++)
        {
            adaptive_search_per_query_result(dataset, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
        }

    }
}

int main() {
    std::cout << "Starting experiments for Shiro-ef hnswdis library...\n\n"
              << std::endl;
    // print the root path
    //  get the root path, if it is not set, immediate exit
    char *root_path = std::getenv("EXPERIMENTS_ROOT");
    if (root_path == nullptr)
    {
        std::cerr << "Error: EXPERIMENTS_ROOT environment variable is not set." << std::endl;
        std::cerr << "Please set it to the root path of the experiments directory." << std::endl;
        std::cerr << "For example, in bash: export EXPERIMENTS_ROOT=/path/to/experiments" << std::endl;
        return 1;
    }
    std::cout << "EXPERIMENTS_ROOT: " << root_path << std::endl;

    // indexing_exp(); // indexes are precomputed, uncomment to run if needed
    // functions for computing groundtruth: compute_groundtruth_laion_text2image and compute_and_save_gound_truth

    offline_exp(true);      // offline computation of estimator, samplings, and ef-adaptor
    online_exp();           // onine search experiments
    // sensitivity_analysis(); // sensitivity analysis for estimator parameters, including k and recall target

    // insert_exp(true); // insert experiment with setup
    // delete_exp(true); // delete experiment with setup

    // ablation_study_visited_list_size();      // ablation study on distance list size
    // ablation_study_sampling_size();           // ablation study on sampling size
    // ablation_study_weighted_decay_function(); // ablation study on weighted decay functions
    // ablation_study_truncation_ratio();
    // ablation_study_n_cv_tables();
    // ablation_study_min_queries_per_score();        // ablation study on truncation ratio

    // per_query_result_exp(); // per-query result experiments
    return 0;
}
