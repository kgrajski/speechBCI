import string
import collections
from tabulate import tabulate


class LabelAnalyzer:
    """
    Analyzes and provides statistics on text labels in a dataset.
    """

    def __init__(
        self,
        labels,
        train_test_flags=None,
    ):
        """Initialize with dataset labels and optional train/test flags."""
        self.labels = labels
        self.val_flag = train_test_flags

        # Calculate overall statistics - FIX: Use self to access the static method
        self.overall_stats = self._get_label_statistics(labels)

        # Calculate train/test stats if flags provided
        if train_test_flags is not None:
            self.train_test_stats = self._train_test_label_compare()

    def get_overall_stats(self):
        """Returns statistics for the entire dataset"""
        return self.overall_stats

    def get_train_test_stats(self):
        """Returns comparative statistics between train and test sets"""
        if not hasattr(self, "train_test_stats"):
            raise ValueError("No train/test flags provided during initialization")
        return self.train_test_stats

    def print_overall_stats(self):
        """Prints overall statistics in a formatted table"""
        stats = {
            k: v
            for k, v in self.overall_stats.items()
            if k not in ["unique_words", "words"]
        }

        print("Dataset Statistics:")
        print(tabulate(stats.items(), headers=["Statistic", "Value"], tablefmt="grid"))

    def print_train_test_comparison(self):
        """Prints the comparison between train and test sets"""
        if not hasattr(self, "train_test_stats"):
            raise ValueError("No train/test flags provided during initialization")
        self._pretty_print_label_stats(self.train_test_stats)

    def _train_test_label_compare(self):
        """Compares label statistics between training and testing sets."""
        # Split labels into training and testing sets
        train_labels = [
            label for label, flag in zip(self.labels, self.val_flag) if not flag
        ]
        test_labels = [label for label, flag in zip(self.labels, self.val_flag) if flag]

        # FIX: Use self to access the static method
        train_stats = self._get_label_statistics(train_labels)
        test_stats = self._get_label_statistics(test_labels)

        unique_train_words = train_stats["unique_words"]
        unique_test_words = test_stats["unique_words"]

        # Calculate common, unique to training, and unique to testing words
        common_words = unique_train_words & unique_test_words
        words_unique_to_train = unique_train_words - unique_test_words
        words_unique_to_test = unique_test_words - unique_train_words

        # For the common_words, what is their frequency in the training and testing sets?
        common_word_counts_train = {
            word: train_stats["words"].count(word) for word in common_words
        }
        common_word_counts_test = {
            word: test_stats["words"].count(word) for word in common_words
        }

        # Get top 20 common words in training and testing sets
        top_20_common_words_train = collections.Counter(
            common_word_counts_train
        ).most_common(20)
        top_20_common_words_test = collections.Counter(
            common_word_counts_test
        ).most_common(20)

        # Get top 20 unique words in training and testing sets
        top_20_unique_train_words = collections.Counter(
            {word: train_stats["words"].count(word) for word in words_unique_to_train}
        ).most_common(20)
        top_20_unique_test_words = collections.Counter(
            {word: test_stats["words"].count(word) for word in words_unique_to_test}
        ).most_common(20)

        label_stats = {
            "train_stats": train_stats,
            "test_stats": test_stats,
            "common_words_count": len(common_words),
            "unique_train_words_count": len(unique_train_words),
            "unique_test_words_count": len(unique_test_words),
            "top_20_common_words_train": top_20_common_words_train,
            "top_20_common_words_test": top_20_common_words_test,
            "top_20_unique_train_words": top_20_unique_train_words,
            "top_20_unique_test_words": top_20_unique_test_words,
        }

        return label_stats

    @staticmethod
    def _get_label_statistics(labels):

        # Remove punctuation and convert to lowercase
        translator = str.maketrans("", "", string.punctuation)
        cleaned_labels = [label.translate(translator).lower() for label in labels]

        # Find the length of the longest and shortest labels
        longest_label_length = len(max(cleaned_labels, key=len))
        shortest_label_length = len(min(cleaned_labels, key=len))

        # Split labels into words
        words = [word for label in cleaned_labels for word in label.split()]

        # Count word frequencies
        word_counts = collections.Counter(words)

        # Get unique words
        unique_words = set(words)

        # Calculate average sentence length
        avg_sentence_length = sum(len(label.split()) for label in cleaned_labels) / len(
            cleaned_labels
        )

        # Get top 20 most frequent words
        top_20_words = word_counts.most_common(20)

        # Get bottom 20 least frequent words
        bottom_20_words = word_counts.most_common()[:-21:-1]

        # Calculate total number of words
        total_words = len(words)

        return {
            "longest_label_length": longest_label_length,
            "shortest_label_length": shortest_label_length,
            "longest_label_length": longest_label_length,
            "shortest_label_length": shortest_label_length,
            "words": words,
            "unique_words": unique_words,
            "unique_words_count": len(unique_words),
            "top_20_words": top_20_words,
            "bottom_20_words": bottom_20_words,
            "avg_sentence_length": avg_sentence_length,
            "total_words": total_words,
        }

    @staticmethod
    def _pretty_print_label_stats(label_stats):
        """
        Pretty prints the contents of the label_stats dictionary.

        Args:
            label_stats (dict): Dictionary containing various statistics and comparisons.
        """
        # Exclude 'unique_words' and 'words' keys from train_stats and test_stats
        train_stats_filtered = {
            k: v
            for k, v in label_stats["train_stats"].items()
            if k not in ["unique_words", "words"]
        }
        test_stats_filtered = {
            k: v
            for k, v in label_stats["test_stats"].items()
            if k not in ["unique_words", "words"]
        }

        print("Training Set Statistics:")
        print(
            tabulate(
                train_stats_filtered.items(),
                headers=["Statistic", "Value"],
                tablefmt="grid",
            )
        )

        print("\nTesting Set Statistics:")
        print(
            tabulate(
                test_stats_filtered.items(),
                headers=["Statistic", "Value"],
                tablefmt="grid",
            )
        )

        print("\nCommon Words Count:", label_stats["common_words_count"])
        print("Unique Train Words Count:", label_stats["unique_train_words_count"])
        print("Unique Test Words Count:", label_stats["unique_test_words_count"])

        print("\nTop 20 Common Words in Training Set:")
        print(
            tabulate(
                label_stats["top_20_common_words_train"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Common Words in Testing Set:")
        print(
            tabulate(
                label_stats["top_20_common_words_test"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Unique Train Words:")
        print(
            tabulate(
                label_stats["top_20_unique_train_words"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        print("\nTop 20 Unique Test Words:")
        print(
            tabulate(
                label_stats["top_20_unique_test_words"],
                headers=["Word", "Count"],
                tablefmt="grid",
            )
        )

        # Only print mask lengths if they exist in the stats
        if "max_embedded_label_len" in label_stats:
            print("\nMax Embedded Label Length:", label_stats["max_embedded_label_len"])
            print("Max Attention Mask Length:", label_stats["max_attention_mask_len"])
