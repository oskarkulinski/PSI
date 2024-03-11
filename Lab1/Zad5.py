def sumDiffProduct(numbers):
    sum = 0
    prod = 1
    diff = 2 * numbers[0]
    
    for i in numbers:
        sum += i
        diff -= i
        prod *= i

    return sum, diff, prod

print(sumDiffProduct([1, 2, 3]))