from collections import defaultdict, Counter
'''
    ToDo:
        - filter out invalid siteswaps
'''

class Siteswap_predictor:
    def __init__(self, max_balls: int):
        self.__max_balls = max_balls


    def __normalize_siteswap(self, siteswap):
        if siteswap is None or len(siteswap) == 0:
            return None
        # Find the maximum value and its indices
        max_value = max(siteswap)
        max_indices = [i for i, value in enumerate(siteswap) if value == max_value]

        # Generate cyclic permutations with the maximum value at the start
        permutations = [siteswap[i:] + siteswap[:i] for i in max_indices]

        # Return the lexicographically biggest permutation
        return max(permutations)


    def __find_minimum_period(self, siteswap: list):
        n = len(siteswap)

        # Check all possible period lengths
        for period in range(1, n // 2 + 1):
            # Check if repeating the prefix generates the full sequence
            if siteswap[:period] * (n // period) == siteswap[:n]:
                return siteswap[:period]

        # If no repeating pattern is found, return the full sequence
        return siteswap
    

    def __trim_siteswap(self, untrimmed_siteswap: list):
        siteswap_set = set(untrimmed_siteswap)
        trimmed_siteswap = []

        for length in range(len(siteswap_set), len(untrimmed_siteswap) // 2 + 1):
            candidate = untrimmed_siteswap[:length]
            test = untrimmed_siteswap[length:length + len(candidate)]
            if candidate == test:
                trimmed_siteswap = candidate
                break

        return trimmed_siteswap
    
    def __adjust_for_2_throws(self, repeating_pattern: list):
        adjusted_pattern = []
        hands_pattern = [item[1] for item in repeating_pattern]
        
        for i in range(len(repeating_pattern)):
            current_hand = hands_pattern[i]
            
            if i > 0 and current_hand == hands_pattern[i - 1]:
                prev_ball = repeating_pattern[i - 2]
                adjusted_pattern.append(prev_ball)

            adjusted_pattern.append(repeating_pattern[i])

        return adjusted_pattern


    def __find_possible_siteswap(self, repeating_pattern: list):
        adjusted_pattern = self.__adjust_for_2_throws(repeating_pattern)
        siteswap = []
        n = len(adjusted_pattern)

        # Extract only the numeric part for Siteswap computation
        balls_pattern = [item[0] for item in adjusted_pattern]

        # Compute Siteswap normally
        for i in range(n):
            current_ball = balls_pattern[i]

            # Find the next occurrence
            for j in range(i + 1, n):
                if balls_pattern[j] == current_ball:
                    siteswap.append(j - i)
                    break

        # print("Untrimmed Siteswap", siteswap)

        trimmed_siteswap = self.__trim_siteswap(siteswap)

        # print("Trimmed Siteswap", trimmed_siteswap)

        return self.__find_minimum_period(trimmed_siteswap)


    def __find_repeating_patterns_with_all_numbers(self, catch_history):
        def find_repeating_patterns(history):
            # Ensure `max_balls` determines the expected set of numbers
            balls_set = set(range(1, self.__max_balls + 1))  # Set of all expected balls
            pattern_counts = defaultdict(int)
            n = len(history)

            # Extract only numbers from the history for repetition checks
            numeric_history = [item[0] for item in history]

            # Check all possible sub-patterns
            for length in range(self.__max_balls, n // 2 + 1):  # Minimum length is `max_balls`
                start = 0
                while start <= n - length:
                    # Define the numeric pattern and check consecutive occurrences
                    numeric_pattern = tuple(numeric_history[start:start + length])
                    occurrences = 1
                    while (
                        start + occurrences * length <= n - length and
                        tuple(numeric_history[start + occurrences * length:start + (occurrences + 1) * length]) == numeric_pattern
                    ):
                        occurrences += 1

                    # If repeated consecutively, count pattern and skip ahead
                    if occurrences > 1:
                        # Extract the full pattern (including metadata) for output
                        full_pattern = tuple(history[start:start + length * occurrences])

                        # Check if the numeric pattern contains all balls
                        numbers_in_pattern = {item[0] for item in full_pattern}
                        if numbers_in_pattern == balls_set:
                            pattern_counts[full_pattern] += occurrences
                            # start += occurrences * length
                            # continue
                    start += 1

            # Convert results into list of dictionaries
            repeating_patterns = [
                {"pattern": list(pattern), "count": count}
                for pattern, count in pattern_counts.items()
            ]

            return repeating_patterns

        repeating_patterns = find_repeating_patterns(catch_history)
        return repeating_patterns


    # def print_siteswaps(self, catch_history):
    #     repeating_patterns = self.__find_repeating_patterns_with_all_numbers(catch_history)

    #     for pattern_dict in repeating_patterns:
    #         # pattern = pattern_dict["pattern"] * pattern_dict["count"]
    #         pattern = pattern_dict["pattern"]
    #         possible_siteswap = self.__find_possible_siteswap(pattern)
    #         normalized_siteswap = self.__normalize_siteswap(possible_siteswap)
    #         print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    #         print("Pattern", pattern)
    #         print("Siteswap", possible_siteswap)
    #         print("Normalized", normalized_siteswap)
    #         print("________________")

    
    def predict_possible_siteswaps(self, catch_history):
        repeating_patterns = self.__find_repeating_patterns_with_all_numbers(catch_history)
        possible_siteswaps = []

        for pattern_dict in repeating_patterns:
            pattern = pattern_dict["pattern"]
            possible_siteswap = self.__find_possible_siteswap(pattern)
            normalized_siteswap = self.__normalize_siteswap(possible_siteswap)
            possible_siteswaps.append(normalized_siteswap)
            # print("---------------------------")
            # print("Pattern", pattern)
            # print("Possible Siteswap", possible_siteswap)
            # print("Normalized Siteswap", normalized_siteswap)

        return possible_siteswaps
    

    def calculate_confidence(self, siteswaps):
        # Represent None and empty elements as "N/A"
        standardized_siteswaps = [
            tuple(s) if s else "N/A" for s in siteswaps
        ]
        
        # Count occurrences of each standardized siteswap
        counts = Counter(standardized_siteswaps)
        total = sum(counts.values())
        
        # Calculate numeric confidence values and sort
        sorted_counts = sorted(
            counts.items(),
            key=lambda item: item[1] / total,  # Sort by confidence value
            reverse=True
        )
        
        # Format percentages after sorting
        percentages = [
            (
                siteswap[0] if isinstance(siteswap, tuple) and len(siteswap) == 1 else siteswap, 
                f"{(count / total) * 100:.2f}%"
            )
            for siteswap, count in sorted_counts
        ]
        
        return percentages


if __name__ == '__main__':
    siteswap_predictor = Siteswap_predictor(3)
    # untrimmed_siteswap = [3, 1, 5, 3, 1, 5, 3, 1, 1]
    # trimmed_siteswap = siteswap_predictor.trim_siteswap(untrimmed_siteswap)
    # print("trimmed siteswap", trimmed_siteswap)
    adjusted = siteswap_predictor.__adjust_for_2_throws([(2, 'right'), (1, 'left'), (3, 'left'), (1, 'right'), (2, 'right'), (1, 'left'), (3, 'left'), (1, 'right')])
    print("Expected", "[(2, 'right'), (1, 'left'), (3, 'left'), (1, 'right'), (2, 'right'), (1, 'left'), (3, 'left'), (1, 'right')]")
    print("Adjusted", adjusted)