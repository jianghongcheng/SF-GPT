def is_bad_version(s):
    return s >= v

def first_bad_version(n):
    first = 1
    last = n
    calls = 0
    while first <= last:  
        mid = first + (last - first) // 2;
        if is_bad_version(mid):  
            last = mid - 1
        else:
            first = mid + 1
        calls += 1
    return [first, calls]

# Driver code
def main():
    global v
    test_case_versions = [1, 2, 3, 4, 5,6,7,8]
    first_bad_versions = [6]

    for i in range(len(test_case_versions)):
        v = first_bad_versions[i]
        if i > 0:
            print()
        print(i + 1,  ".\tNumber of versions: ", test_case_versions[i], sep="")
        result = first_bad_version(test_case_versions[i])
        print("\n\tFirst bad version: ",
              result[0], ". Found in ", result[1], " API calls.", sep="")
        print("-"*100)


if __name__ == '__main__':
    main()