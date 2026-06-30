#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../hnswlib/adaptive_ef.h" // Includes all necessary headers

using namespace std;

size_t estimate_ef_for_target(
    const vector<pair<int, vector<pair<int, float>>>>& estimators, 
    float score, 
    float target_recall) 
{
    if (estimators.empty()) return 0;
    
    auto entry = lower_bound(estimators.begin(), estimators.end(), score, 
        [](const auto &a, const float &b) { return a.first < b; });

    if (entry == estimators.begin()) {
        entry = estimators.begin();
    } else if (entry == estimators.end()) {
        entry = prev(estimators.end());
    }

    for (const auto &ef_recall : entry->second) {
        if (ef_recall.second >= target_recall) {
            return ef_recall.first;
        }
    }
    return entry->second.back().first;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <path_to_ef_adaptor_bin> <target_recall>" << endl;
        cerr << "Example: " << argv[0] << " ../estimation_table/deep-image-96-angular-ef_adaptor--k100-ef.bin 0.95" << endl;
        return 1;
    }

    string bin_path = argv[1];
    float target_recall = stof(argv[2]);

    cout << "Loading estimation table from: " << bin_path << endl;
    try {
        hnswdis::EfAdapter adapter(bin_path);
        
        cout << "Target Recall: " << target_recall << endl;
        cout << "Original expected recall in file: " << adapter.get_expected_recall() << endl;
        cout << string(50, '-') << endl;

        if (adapter.has_dep_tables()) {
            const auto& dep_tables = adapter.get_all_tables();
            const auto& dep_thresholds = adapter.get_dep_thresholds();
            cout << "Found 2D Lookup Table with " << dep_tables.size() << " d_ep buckets." << endl;
            
            for (size_t t = 0; t < dep_tables.size(); ++t) {
                float lower_bound = (t == 0) ? 0.0f : dep_thresholds[t-1];
                float upper_bound = (t == dep_thresholds.size()) ? std::numeric_limits<float>::infinity() : dep_thresholds[t];
                
                cout << "\n=== Bucket " << t << " | d_ep range: [" << lower_bound << ", " << upper_bound << ") ===" << endl;
                cout << left << setw(10) << "Score" << setw(15) << "Estimated EF" << "Available (EF, Recall) distribution in group" << endl;
                cout << string(80, '-') << endl;
                
                for (const auto& group : dep_tables[t]) {
                    int score = group.first;
                    size_t estimated_ef = estimate_ef_for_target(dep_tables[t], score, target_recall);
                    cout << left << setw(10) << score << setw(15) << estimated_ef;
                    int count = 0;
                    for (const auto& ef_rec : group.second) {
                        if (count >= 50) { cout << "..."; break; }
                        cout << "[" << ef_rec.first << ":" << fixed << setprecision(3) << ef_rec.second << "] ";
                        count++;
                    }
                    cout << endl;
                }
            }
        } else {
            const auto& estimators = adapter.get_ef_recall_estimators();
            cout << "Loaded 1D Lookup Table with " << estimators.size() << " score groups." << endl;
            cout << left << setw(10) << "Score" << setw(15) << "Estimated EF" << "Available (EF, Recall) distribution in group" << endl;
            cout << string(80, '-') << endl;

            for (const auto& group : estimators) {
                int score = group.first;
                size_t estimated_ef = estimate_ef_for_target(estimators, score, target_recall);
                cout << left << setw(10) << score << setw(15) << estimated_ef;
                int count = 0;
                for (const auto& ef_rec : group.second) {
                    if (count >= 50) { cout << "..."; break; }
                    cout << "[" << ef_rec.first << ":" << fixed << setprecision(3) << ef_rec.second << "] ";
                    count++;
                }
                cout << endl;
            }
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
