#!/bin/bash

# count_sentence_fields_awk.sh - More efficient version using awk

# Process each file containing "sentence" in its name
find . -maxdepth 1 -type f -name "*sentence*" | while read -r file; do
    # Extract filename from path for cleaner output
    filename=$(basename "$file")
    
    # Use awk to check for 6th line and count fields in one operation
    result=$(awk 'NR==6 {print NF; exit} END {if (NR<6) print "ERROR"}' "$file")
    
    if [ "$result" = "ERROR" ]; then
        echo "$filename: Has fewer than 6 lines"
    else
        echo "$filename: $result fields"
    fi
done

