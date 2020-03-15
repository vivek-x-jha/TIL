def twoSum(nums, target):
    complements = {}
    for i, num in enumerate(nums):
        comp = target - num
        if comp not in complements:
            complements[num] = i
        else:
            assert target == nums[complements[comp]] + nums[i]
            return [complements[comp], i]


nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums, target))
