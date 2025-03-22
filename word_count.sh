#!/bin/bash
# File: count_unique_words.sh
# Description: Counts unique words in original predictions and creates a plot

# Set the target directory (default to current directory if not specified)
TARGET_DIR="${1:-.}"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Check if gnuplot is installed
if ! command -v gnuplot &> /dev/null; then
    echo "Error: gnuplot is not installed. Please install it with:"
    echo "sudo apt-get install gnuplot"
    exit 1
fi

echo "Processing prediction files in: $TARGET_DIR"
echo "----------------------------------------"

# Create temporary files for sorting and plotting data
TEMP_FILE=$(mktemp)
PLOT_DATA=$(mktemp)

# Find all .txt files and extract the 9th field for sorting
find "$TARGET_DIR" -maxdepth 1 -name "*.txt" | while read file; do
    filename=$(basename "$file")
    # Extract the 9th field (index 8) by splitting on underscore
    sort_key=$(echo "$filename" | awk -F'_' '{print $9}')
    # Only include files where we can extract a numeric 9th field
    if [[ "$sort_key" =~ ^[0-9]+$ ]]; then
        echo "$sort_key $file" >> "$TEMP_FILE"
    fi
done

# Header for results table
printf "%-50s %8s\n" "FILENAME" "UNIQUE WORDS"
printf "%-50s %8s\n" "$(printf '%0.s-' {1..50})" "$(printf '%0.s-' {1..8})"

# Sort files numerically by the extracted field and collect data for plotting
cat "$TEMP_FILE" | sort -n | while read sort_key file; do
    # Check if file exists (just to be safe)
    if [ -f "$file" ]; then
        # Get just the filename without the path
        filename=$(basename "$file")
        
        # Run the word counting command
        word_stats=$(cat "$file" | grep "Predict" | grep "original" | \
                    sort | uniq -c | awk -F':' '{print $2}' | tr ' ' '\n' | \
                    sort | uniq | wc -l)
        
        # Display the result
        printf "%-50s %8d\n" "$filename" "$word_stats"
        
        # Save data for plotting
        echo "$sort_key $word_stats" >> "$PLOT_DATA"
    fi
done

echo "----------------------------------------"

# Create gnuplot script
GNUPLOT_SCRIPT=$(mktemp)
cat > "$GNUPLOT_SCRIPT" << EOL
set terminal pngcairo enhanced font "arial,12" size 800,600
set output 'word_count_plot.png'
set title 'Unique Words per Epoch'
set xlabel 'Epoch Number'
set ylabel 'Unique Word Count'
set grid
set key off
set style data linespoints
set pointsize 1.5
plot '$PLOT_DATA' using 1:2 with linespoints linecolor rgb "#0060ad" linewidth 2
EOL

# Run gnuplot to create the graph
gnuplot "$GNUPLOT_SCRIPT"

# Clean up temporary files
rm -f "$TEMP_FILE" "$PLOT_DATA" "$GNUPLOT_SCRIPT"

echo "Plot created: word_count_plot.png"
echo "Note: This script counts unique words in original predictions for each file"
echo ""
echo "Usage: ./count_unique_words.sh [directory_path]"
echo "If no directory is specified, the current directory is used."