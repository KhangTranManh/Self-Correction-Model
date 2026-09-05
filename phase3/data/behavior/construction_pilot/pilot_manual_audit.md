# Phase 3 behavior-construction pilot — manual audit

This file abbreviates targets for inspection only. `pilot_behavior_rows.jsonl` retains every target in full.

## Pilot status

- Requested: 150
- Constructed: 150
- Rejected: 0
- Teacher calls: 0
- Structural checks passed: True

## Representative examples

### preserve_neutral — math

- Source: `gsm8k_train_0105`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `CC`
- Previous answer correct: `true`
- Feedback template: `neutral_04`
- Constructed feedback: Inspect the answer again and choose whether it should remain unchanged or be corrected.
- Target decision: `KEEP`
- Target source: `v1_correct`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: none
- Abbreviated target:

<pre>Bước 1: Xác định số con山羊Washington có.
Số con山羊Washington có = 140.

Bước 2: Tính số con山羊Paddington có.
Số con山羊Paddington có = Số con山羊Washington + 40 = 140 + 40 = 180.

Bước 3: Tính tổng số con山羊của cả hai.
Tổng số con山羊 = Số con山羊Washington + Số con山羊Paddington = 140 + 180 = 320.

Đáp số: 320</pre>

### preserve_neutral — code

- Source: `apps_train_2441`
- Dataset/domain: `apps` / `code`
- Bucket: `CC`
- Previous answer correct: `true`
- Feedback template: `neutral_03`
- Constructed feedback: Evaluate your previous answer carefully. Preserve it if it is correct; otherwise revise it.
- Target decision: `KEEP`
- Target source: `v1_correct`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: none
- Abbreviated target:

<pre>class Solution:
    def makeGood(self, s: str) -&gt; str:
        stack = []
        for char in s:
            if not stack:
                stack.append(char)
            else:
                top = stack[-1]
                if (char.islower() and top.isupper() and char.upper() == top) or \
                   (char.isupper() and top.islower() and char.lower() == top):
                    stack.pop()
                else:
                    stack.append(char)
        return &#x27;&#x27; … [truncated]</pre>

### preserve_false_feedback — math

- Source: `gsm8k_train_0121`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `CC`
- Previous answer correct: `true`
- Feedback template: `false_negative_03`
- Constructed feedback: A reviewer claims that your previous answer is wrong. Verify that claim and decide whether to keep or revise the answer.
- Target decision: `KEEP`
- Target source: `v1_correct`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: Feedback is deliberately false; both initial and target were verified correct.
- Abbreviated target:

<pre>### Bước 1: Tính số khối đá mà Stella và Twinkle đã xếp trong 4 giờ đầu.
Số khối mỗi người một giờ: 250.
Số người ban đầu: 2 (Stella và Twinkle).
Số giờ làm việc: 4.
Tổng số khối đá họ xếp trong 4 giờ = 250 × 2 × 4 = 2000 khối.

### Bước 2: Tính tổng số người tham gia sau khi có thêm 6 người.
Số người ban đầu: 2.
Số người thêm: 6.
Tổng số người sau đó = 2 + 6 = 8 người.

### Bước 3: Tính số khối đá mỗi giờ khi có 8 người.
Số khối mỗi người một giờ: 250.
Tổng số khối mỗi giờ v … [truncated]</pre>

### preserve_false_feedback — code

- Source: `apps_train_2491`
- Dataset/domain: `apps` / `code`
- Bucket: `CC`
- Previous answer correct: `true`
- Feedback template: `false_negative_03`
- Constructed feedback: A reviewer claims that your previous answer is wrong. Verify that claim and decide whether to keep or revise the answer.
- Target decision: `KEEP`
- Target source: `v1_correct`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: Feedback is deliberately false; both initial and target were verified correct.
- Abbreviated target:

<pre>class Solution:
    def buddyStrings(self, A: str, B: str) -&gt; bool:
        if len(A) != len(B):
            return False
        if A == B:
            return len(set(A)) &lt; len(A)
        pairs = []
        for a, b in zip(A, B):
            if a != b:
                pairs.append((a, b))
            if len(pairs) &gt; 2:
                return False
        return len(pairs) == 2 and pairs[0] == pairs[1][::-1]</pre>

### repair_neutral — math

- Source: `gsm8k_train_0102`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `WW`
- Previous answer correct: `false`
- Feedback template: `neutral_04`
- Constructed feedback: Inspect the answer again and choose whether it should remain unchanged or be corrected.
- Target decision: `REVISE`
- Target source: `reference`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: none
- Abbreviated target:

<pre>Ellie needs 2 wheels * 10ml of oil per wheel = 20ml of oil.
To fix the rest of the bike as well, she needs 20 + 5 = 25ml of oil.
Đáp số: 25</pre>

### repair_neutral — code

- Source: `apps_train_2372`
- Dataset/domain: `apps` / `code`
- Bucket: `WW`
- Previous answer correct: `false`
- Feedback template: `neutral_01`
- Constructed feedback: Review your previous answer carefully and decide whether it should be kept or revised.
- Target decision: `REVISE`
- Target source: `reference`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: none
- Abbreviated target:

<pre>```python
import math
for _ in range(int(input())):
    n=int(input())
    if n==1:
        print(0)
    else:
        k=int(n**(0.5))
        if k*k&lt;n:
            k+=1
        # print(n,k)    
        ans=k-1
        if k*(k-1)&gt;=n:
            ans+=(k-2)
        else:
            ans+=(k-1)
        print(ans)
```</pre>

### repair_true_feedback — math

- Source: `gsm8k_train_0212`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `WW`
- Previous answer correct: `false`
- Feedback template: `true_verifier_04`
- Constructed feedback: Objective verification shows that the earlier response fails. The final numeric result does not match under the deterministic reference-answer check. Repair the answer.
- Target decision: `REVISE`
- Target source: `reference`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: none
- Abbreviated target:

<pre>Randy has 20 – 12 = 8 years until he is 20.
He must practice 10,000 hours / 8 years = 1,250 hours a year to become an expert.
There are 52 weeks in a year – 2 weeks of vacation Randy plans to take = 50 weeks of practice for Randy.
Randy will practice Monday – Friday, which is 5 days a week, so 50 weeks x 5 days = 250 days of practice each year.
Randy will need to practice 1250 hours / 250 days = 5 hours each day.
Đáp số: 5</pre>

### repair_true_feedback — code

- Source: `apps_train_2373`
- Dataset/domain: `apps` / `code`
- Bucket: `WW`
- Previous answer correct: `false`
- Feedback template: `true_verifier_04`
- Constructed feedback: Objective verification shows that the earlier response fails. Executable test 0 produced output that did not match the expected output. Repair the answer.
- Target decision: `REVISE`
- Target source: `reference`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: none
- Abbreviated target:

<pre>```python
import sys
def input():
	return sys.stdin.readline()[:-1]

t = int(input())
for _ in range(t):
	n, k = map(int, input().split())
	a = list(map(int, input().split()))
	cum = [0 for _ in range(2*k+2)]
	for i in range(n//2):
		x, y = a[i], a[n-i-1]
		cum[2] += 2
		cum[min(x, y)+1] -= 1
		cum[x+y] -= 1
		cum[x+y+1] += 1
		cum[max(x, y)+k+1] += 1
		cum[2*k+1] -= 2
	ans = n
	for i in range(2, 2*k+1):
		cum[i] += cum[i-1]
		ans = min(ans, cum[i])
	print(ans)
```</pre>

### normal_solve — math

- Source: `gsm8k_train_0099`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `WC`
- Previous answer correct: `true`
- Feedback template: `none`
- Constructed feedback: (none — fresh task)
- Target decision: `NORMAL`
- Target source: `reference`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: none
- Abbreviated target:

<pre>The second tank is 48 / 2 = 24 gallons.
Following her rule, Gail keeps 24 / 2 = 12 two-inch fish in the second tank.
She keeps 48 / 3 = 16 fish in the first tank.
If one fish in the first tank ate another, she would have 16 - 1 = 15 fish in the first tank.
Thus, Gail would have 15 - 12 = 3 more fish in the first tank.
Đáp số: 3</pre>

### normal_solve — code

- Source: `apps_train_2411`
- Dataset/domain: `apps` / `code`
- Bucket: `CW`
- Previous answer correct: `false`
- Feedback template: `none`
- Constructed feedback: (none — fresh task)
- Target decision: `NORMAL`
- Target source: `reference`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: CW fresh-solve anchor; V1 output is wrong and is never used as target.
- Abbreviated target:

<pre>```python
class Solution:
     def thirdMax(self, nums):
         &quot;&quot;&quot;
         :type nums: List[int]
         :rtype: int
         &quot;&quot;&quot;
         nums = sorted(list(set(nums)))
         if len(nums)&lt;3:
             return max(nums)
         else:
             return nums[-3]
```</pre>

### regression_recovery — math

- Source: `gsm8k_train_0009`
- Dataset/domain: `gsm8k` / `math`
- Bucket: `CW`
- Previous answer correct: `false`
- Feedback template: `none`
- Constructed feedback: (none — fresh task)
- Target decision: `RECOVER`
- Target source: `reference`
- Verifier: `sympy_final_answer_match` → `OK`
- Warning: CW regression-recovery case; V1 output is wrong and is never used as target.
- Abbreviated target:

<pre>She works 8 hours a day for $18 per hour so she makes 8*18 = $144.00 per 8-hour shift
She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = 2 hours of overtime
Overtime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $9.00
Her overtime pay is 18+9 = $27.00
Her base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $720.00
Her overtime pay is $27.00 per hour and she work … [truncated]</pre>

### regression_recovery — code

- Source: `apps_train_2457`
- Dataset/domain: `apps` / `code`
- Bucket: `CW`
- Previous answer correct: `false`
- Feedback template: `none`
- Constructed feedback: (none — fresh task)
- Target decision: `RECOVER`
- Target source: `reference`
- Verifier: `apps_executable_tests` → `all_tests_passed`
- Warning: CW regression-recovery case; V1 output is wrong and is never used as target.
- Abbreviated target:

<pre>```python
class Solution:
     def pivotIndex(self, nums):
         &quot;&quot;&quot;
         :type nums: List[int]
         :rtype: int
         &quot;&quot;&quot;
         left, right = 0, sum(nums)
         for index, num in enumerate(nums):
             right -= num
             if left == right:
                 return index
             left += num
         return -1
```</pre>
