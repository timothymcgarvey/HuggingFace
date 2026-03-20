from collections import defaultdict

def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    seen = defaultdict(int)
    seen[0] = 1  # important edge case

    for num in nums:
        prefix_sum += num
        count += seen[prefix_sum - k]
        seen[prefix_sum] += 1

    return count
if __name__ == "__main__":
    ar=[1,2,3,4,5]
    k=5
    ss=subarray_sum(ar,k)
    print(ss)