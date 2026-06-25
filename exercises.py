"""

0 1 2 3 4 5
1 0 0 0 0 0
2 0 0 0 0 0 
3 0 0 0 0 0
4 0 0 0 0 0 

abmsd
cdlad

at zero position we don't have any edits. 
The row # 1 means that for each column in this row we need to make c edits to get the word up to the length c
The same workds for col # 1 but rows.
now, at any position if the letters are the same, take the value on the diagonal (first value is 0)
if letters are different take 1 + min(up, diagonal, left). 

Target is row, source is column.
Column 1 represents deletions from source because target has zero index at this column. 
First row represents insertions to source because the target is represented by the row

"""


def min_edit_distance(s1, s2 ):
    m = len(s1)  # 6
    n = len(s2)
    dp = [
        [i + j if i == 0 or j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)
    ]

    """
    s1[0] vs s2[0], s2[1], etc.
    """

    if not s1 and not s2:
        return 0
    
    if not s1 or not s2:
        return len(s1) or len(s2)


    for ch1 in range(1, m+1):
        for ch2 in range(1, n+1):
            if s1[ch1 - 1] == s2[ch2 - 1]:
                dp[ch1][ch2] = dp[ch1-1][ch2-1]
            else:
                dp[ch1][ch2] = 1 + min(dp[ch1-1][ch2-1], dp[ch1-1][ch2], dp[ch1][ch2-1])

    print(dp)
    return dp[-1][-1]




if __name__ == "__main__":
    word1 = "john"
    word2 = "songh"
    levenstein = min_edit_distance(word1, word2)
    print(levenstein)

    def test(a,b):
        return len(a) or len(b)

    a = ""
    b = "kk"

    print(test(a,b))
