import re

with open('research/method1_explanation.tex', 'r') as f:
    text = f.read()

start_scatter = text.find('% Background Shade for Intersection')
end_scatter = text.find('% Legend', start_scatter)

if start_scatter != -1 and end_scatter != -1:
    new_scatter = r"""% Background Shade for Intersection
        \begin{scope}[on background layer]
            \fill[red!10] (0,0) rectangle (1.0, 1.0);
            \node[text=red!80!black, font=\large\bfseries] at (0.5, 0.5) {$\mathcal{H}$};
        \end{scope}

        % Grid
        \foreach \y in {1, 2, 3, 4, 5, 6} \draw[gridline] (0,\y) -- (9,\y);
        \foreach \x in {1, 2, 3, 4, 5, 6, 7, 8} \draw[gridline] (\x,0) -- (\x,6);

        % Axes
        \draw[axis] (0, 0) -- (9.5, 0) node[right] {Coefficient of Variation (CV)};
        \draw[axis] (0, 0) -- (0, 6.5) node[above] {Convergence};
        
        % Percentile Threshold Lines
        \draw[dashed, ultra thick, red!80!black] (1.0, 0) node[below] {$cv^{(5\%)}$} -- (1.0, 6.2);
        \draw[dashed, ultra thick, red!80!black] (0, 1.0) node[left] {$conv^{(5\%)}$} -- (9.2, 1.0);
    
        % Generate Easy Pool (Scattered randomly)
        \foreach \i in {1,...,200} {
            \pgfmathsetmacro{\x}{1.0 + rnd*8.0}
            \pgfmathsetmacro{\y}{1.0 + rnd*5.0}
            \node[easy] at (\x, \y) {};
        }
        
        % 1D Hard only (Bottom Right)
        \foreach \i in {1,...,25} {
            \pgfmathsetmacro{\x}{1.0 + rnd*8.0}
            \pgfmathsetmacro{\y}{rnd*0.95 + 0.02}
            \node[easy] at (\x, \y) {};
        }
        
        % 1D Hard only (Top Left)
        \foreach \i in {1,...,25} {
            \pgfmathsetmacro{\x}{rnd*0.95 + 0.02}
            \pgfmathsetmacro{\y}{1.0 + rnd*5.0}
            \node[easy] at (\x, \y) {};
        }
        
        % 2D Hard Intersection (Guaranteed inclusion, Bottom Left)
        \foreach \i in {1,...,15} {
            \pgfmathsetmacro{\x}{rnd*0.95 + 0.02}
            \pgfmathsetmacro{\y}{rnd*0.95 + 0.02}
            \node[hard, inner sep=0.8pt] at (\x, \y) {};
        }
        
        % Selected Easy (Backfill sampling, Blue) distributed across all non-hard regions
        % 1. Top-Right quadrant
        \foreach \i in {1,...,12} {
            \pgfmathsetmacro{\x}{1.0 + rnd*8.0}
            \pgfmathsetmacro{\y}{1.0 + rnd*5.0}
            \node[selected] at (\x, \y) {};
        }
        % 2. Bottom-Right quadrant (1D hard)
        \foreach \i in {1,...,4} {
            \pgfmathsetmacro{\x}{1.0 + rnd*8.0}
            \pgfmathsetmacro{\y}{rnd*0.95 + 0.02}
            \node[selected] at (\x, \y) {};
        }
        % 3. Top-Left quadrant (1D hard)
        \foreach \i in {1,...,4} {
            \pgfmathsetmacro{\x}{rnd*0.95 + 0.02}
            \pgfmathsetmacro{\y}{1.0 + rnd*5.0}
            \node[selected] at (\x, \y) {};
        }

        """
    text = text[:start_scatter] + new_scatter + text[end_scatter:]

# Add the Pool Reduction Subfigure after the CDF one
start_cdf_end = text.find('\\end{subfigure}', text.find('\\label{fig:cdf}'))
if start_cdf_end != -1:
    pool_viz = r"""
    \vspace{1.5em}

    % --- Middle Figure: Pool Reduction Visualization ---
    \begin{subfigure}[b]{0.85\textwidth}
        \centering
        \begin{tikzpicture}[scale=1.0, axis/.style={thick, ->}]
            % Left side: Massive Pool
            \node[font=\bfseries, below=0.5cm] at (2, -1) {\small $\mathcal{P}$ (Initial Pool)};
            \draw[axis] (0, 0) -- (4, 0);
            \draw[axis] (0, 0) -- (0, 3);
            \node[anchor=north west, font=\footnotesize] at (0,0) {0};
            \node[anchor=north, font=\footnotesize] at (2,0) {$N_{pool} = 30,000$};
            
            % Randomly scatter many dots to represent 30k
            \foreach \i in {1,...,300} {
                \pgfmathsetmacro{\x}{rnd*3.8}
                \pgfmathsetmacro{\y}{rnd*2.8}
                \node[easy] at (\x, \y) {};
            }
            
            % Right side: Target Sample
            \node[font=\bfseries, below=0.5cm] at (7, -1) {\small $\mathcal{S}$ (Target Sample)};
            \draw[axis] (5, 0) -- (9, 0);
            \draw[axis] (5, 0) -- (5, 3);
            \node[anchor=north west, font=\footnotesize] at (5,0) {0};
            \node[anchor=north, font=\footnotesize] at (7,0) {$N_{sample} = 3,000$};
            
            % Scatter fewer dots
            \foreach \i in {1,...,30} {
                \pgfmathsetmacro{\x}{5.1 + rnd*3.8}
                \pgfmathsetmacro{\y}{rnd*2.8}
                \node[easy] at (\x, \y) {};
            }
            
            % Big Arrow
            \draw[ultra thick, -{Stealth[length=3mm]}, blue] (4, 1.5) -- (5, 1.5) node[midway, above, font=\footnotesize] {Hard-First Sampling};
        \end{tikzpicture}
        \caption{The algorithmic pool reduction. Method 1 begins by probing a massive $N_{pool}=30,000$ random vectors to calculate CV and Convergence metrics. From this vast space, it extracts the intersection of the bottom 5\% tails (the Hard Set) and backfills the remaining budget up to $N_{sample}=3,000$.}
        \label{fig:pool_reduction}
    \end{subfigure}
    """
    text = text[:start_cdf_end] + pool_viz + text[start_cdf_end:]

with open('research/method1_explanation.tex', 'w') as f:
    f.write(text)
