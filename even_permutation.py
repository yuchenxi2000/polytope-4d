import itertools


def is_even_permutation(p):
    """
    judge if p is an even permutation
    from the remainder of inversions divided by 2
    """
    n = len(p)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inversions += 1
    return inversions % 2 == 0


def get_even_permutations(lst):
    """
    :return: all even permutations of lst
    """
    all_permutations = itertools.permutations(lst)
    even_permutations = [p for p in all_permutations if is_even_permutation(p)]
    return even_permutations


# generate all even permutations
lst = [0, 1, 2, 3]
even_perms = get_even_permutations(lst)
for perm in even_perms:
    print(perm)
