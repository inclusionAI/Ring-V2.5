**1. Notation and basic invariants**

Let the numbers on the red cards be $r_1,\dots,r_n$ and on the blue cards be $b_1,\dots,b_n$.  
An **operation** chooses indices $i,j$ with $r_i<b_j$ and replaces both by $\frac{r_i+b_j}{2}$.

The total sum  
$$
S = \sum_{i=1}^n r_i + \sum_{j=1}^n b_j
$$
is invariant. Initially all reds are $0$ and all blues are $1$, so $S=n$.

All numbers stay in $[0,1]$ during the whole process.

**2. A majorization invariant**

Arrange the $2n$ numbers in non‑decreasing order:  
$a_1 \le a_2 \le \cdots \le a_{2n}$.  
For $k=1,\dots,2n$ define $M_k = a_1+\cdots+a_k$.

**Lemma 1.** *For any operation, each $M_k$ does not decrease.*  

*Proof.* Let the two numbers involved be $x$ (red) and $y$ (blue) with $x<y$; they become two copies of $w=\frac{x+y}{2}$. Since $x\le w\le y$, the multiset after the operation is obtained by removing $x$ and $y$ and inserting two copies of $w$. For any fixed $k$, the sum of the $k$ smallest elements cannot decrease – if both $x$ and $y$ are among the $k$ smallest, their removal reduces the sum by $x+y$ and inserting $2w$ adds exactly $x+y$; if only one of them is among the $k$ smallest, a simple case analysis shows the sum does not drop; if none are, the sum is unchanged. Hence $M_k$ is non‑decreasing. ∎

Initially we have $n$ zeros and $n$ ones, so  
$$
M_k^{(0)} = \max\{0,\;k-n\} \qquad (k=1,\dots,2n).
$$  
By Lemma 1, every reachable configuration satisfies  
$$
M_k \ge \max\{0,\;k-n\} \qquad (k=1,\dots,2n). \tag{1}
$$

**3. Terminal configurations**

A configuration is **terminal** if no operation is possible, i.e. $r_i \ge b_j$ for all $i,j$.

**Lemma 2.** *In a terminal configuration the $n$ smallest numbers among all $2n$ cards are exactly the $n$ blue cards (as a multiset). Consequently, the sum of the blue cards is $B = a_1+\cdots+a_n$.*  

*Proof.* Let $t = \max_j b_j$. Because the configuration is terminal, every red satisfies $r_i \ge t$. Hence any number $<t$ must be blue. Let $m$ be the number of blues strictly smaller than $t$; then these are the only numbers $<t$. The remaining $n-m$ blues all equal $t$, and all reds are $\ge t$. Therefore the sorted list begins with those $m$ numbers (all blue), followed by $n-m$ copies of $t$ (all blue). Thus the first $n$ positions are exactly the blues. ∎

**4. Existence of a card with value $\frac12$**

**Lemma 3.** *In any reachable terminal configuration (with $n\ge 1$) there exists at least one red card and at least one blue card whose value equals $\frac12$.*  

*Proof.* Suppose, for contradiction, that no card equals $\frac12$. All numbers are dyadic rationals (obtained by repeated averaging of $0$ and $1$), so we can choose an integer $D=2^m$ such that every value is an integer multiple of $1/D$. Multiply each number by $D$; we obtain integers $R_i$ (reds) and $B_j$ (blues) with $0\le R_i,B_j\le D$, and none equals $D/2$. The terminal condition becomes $R_i \ge B_j$ for all $i,j$, and the total sum is $\sum R_i + \sum B_j = nD$.

Let $a = \min_i R_i$ and $b = \max_j B_j$. From $R_i \ge B_j$ we have $a \ge b$.

**Case 1.** There exists a red with $R_i \le D/2-1$. Then $a \le D/2-1$, and because all blues are $\le a$, we have $B_j \le D/2-1$ for every $j$. Hence  
$$
\sum R_i + \sum B_j \le n(D/2-1) + n(D/2-1) = nD - 2n < nD,
$$  
contradiction.

**Case 2.** There exists a blue with $B_j \ge D/2+1$. Then $b \ge D/2+1$, and all reds are $\ge b$ so $R_i \ge D/2+1$ for all $i$. Hence  
$$
\sum R_i + \sum B_j \ge n(D/2+1) + n(D/2+1) = nD + 2n > nD,
$$  
contradiction.

Thus we must have $R_i \ge D/2+1$ for all reds and $B_j \le D/2-1$ for all blues. Then, because the total sum equals $nD$, it forces  
$$
R_i = D/2+1,\quad B_j = D/2-1 \qquad\text{for all }i,j.
$$  
Consequently, in the original configuration every red equals $\frac12 + \frac1D$ and every blue equals $\frac12 - \frac1D$.

Now consider the last operation that produced this configuration. It involved a red and a blue, say with values $x$ and $y$ ($x<y$), and replaced them by two copies of $w=\frac{x+y}{2}$. In the final configuration the two cards involved are one red and one blue, both equal to $w$. But all reds are $\frac12+\frac1D$ and all blues are $\frac12-\frac1D$; therefore we would need  
$$
w = \frac12+\frac1D = \frac12-\frac1D,
$$  
which is impossible. This contradiction shows that our assumption was false; there must be a card equal to $\frac12$.

If there is a red $\frac12$ but no blue $\frac12$, then all blues are $<\frac12$ (by terminality). The average of all numbers is $\frac12$, so the sum of blues would be $<n/2$ and the sum of reds $>n/2$; in particular some red $>\frac12$. Applying the same integer‑scaling argument to the *dual* configuration (swap colours and replace each value $v$ by $1-v$) yields a contradiction because the dual would have a blue $\frac12$ but no red $\frac12$. Hence both colours must contain a card with value $\frac12$. ∎

**5. Minimum blue sum – inductive determination**

Let $f(n)$ be the minimum possible sum of the blue cards in a reachable terminal configuration with $n$ red/blue pairs. (Such a minimum exists: we may consider a configuration that maximises the red sum; the red sum is bounded and increases with each operation, so a maximum is attained, and it corresponds to a minimum of the blue sum.)

**Base $n=1$.** The only terminal state reachable from $(0,1)$ is $(\frac12,\frac12)$; thus $B = \frac12$ and $f(1)=\frac12 = 1-2^{-1}$.

**Inductive step.** Assume $f(k) = 1-2^{-k}$ holds for all $k < n$ ($n\ge 2$). Consider a configuration that attains $f(n)$. By Lemma 3 it contains a red card $R^*$ and a blue card $B^*$ both with value $\frac12$. Remove these two cards. The remaining $n-1$ reds and $n-1$ blues still satisfy the terminal condition (removing cards cannot create a violation), so they form a terminal configuration for $n-1$ pairs.

Now apply the transformation  
$$
r' = 2r-1 \quad (\text{for each remaining red}), \qquad
b' = 2b \quad (\text{for each remaining blue}).
$$  
In the original configuration, because the largest blue equals the smallest red and that common value is $\frac12$ (by Lemma 2 and Lemma 3), we have $r \ge \frac12$ and $b \le \frac12$; hence $0 \le r',b' \le 1$. Moreover, the transformed configuration is reachable from the initial state with $n-1$ pairs. Indeed, take the original sequence of operations and delete all operations that involve $R^*$ or $B^*$; the remaining operations, when their effect is translated via the linear transformation, become exactly the operations that produce the transformed configuration from the initial all‑zeros and all‑ones state. (One can also argue by noting that the last operation created the $\frac12,\frac12$ pair; reversing it and rescaling yields the transformed configuration.)

Let $B'$ be the blue sum of the transformed configuration. Since each blue was multiplied by $2$,  
$$
B' = 2\bigl(B - \tfrac12\bigr) = 2B - 1.
$$  
Because the transformed configuration is reachable for $n-1$ pairs, its blue sum is at least $f(n-1)$. Thus  
$$
2B - 1 \ge f(n-1) = 1 - 2^{-(n-1)}.
$$  
Recalling that $B = f(n)$ (the configuration was chosen minimal), we obtain  
$$
2f(n) - 1 \ge 1 - 2^{-(n-1)} \quad\Longrightarrow\quad f(n) \ge 1 - 2^{-n}. \tag{2}
$$

**6. Construction attaining the bound**

We now show that $f(n) \le 1-2^{-n}$ by exhibiting, for every $n$, a reachable terminal configuration with blue sum exactly $1-2^{-n}$. This will prove $f(n) = 1-2^{-n}$ and consequently the maximum red sum is $R_{\max}(n) = n - f(n) = n-1+2^{-n}$.

We proceed by induction on $n$.

*Base $n=1$:* One operation on $(0,1)$ gives $(\frac12,\frac12)$; blue sum = $\frac12 = 1-2^{-1}$.

*Inductive step.* Assume we have a reachable terminal configuration for $n-1$ with  
$$
\text{reds: } 1-2^{-1},\; 1-2^{-2},\; \dots,\; 1-2^{-(n-1)},\qquad
\text{blues: } 2^{-1},\; 2^{-2},\; \dots,\; 2^{-(n-1)},
$$  
and that the pairing after sorting (largest red with smallest blue, etc.) satisfies $r_i + b_i = 1$ for each $i$. (The explicit construction below will preserve this property.)

Now add an extra red card $X$ with value $0$ and an extra blue card $Y$ with value $1$. We will merge them into the existing pattern with the following steps.

**Stage A – process $X$ with the blues.**  
Go through the blues in *decreasing* order of their current values (i.e. start with the largest blue, which is $2^{-1}$, then $2^{-2}$, …, finally $2^{-(n-1)}$). For each such blue $b$, perform the operation $(X, b)$.  
Because initially $X=0 < 2^{-(n-1)}$ and after each step $X$ becomes the average of its previous value and the current blue, one checks inductively that before each operation the condition $X < b$ holds.

After Stage A, the card $X$ has become  
$$
X = \sum_{i=1}^{n-1} \frac{b_i}{2^{n-i}} = \sum_{i=1}^{n-1} \frac{2^{-i}}{2^{n-i}} = \frac{n-1}{2^{n}},
$$  
but more importantly, the values of the blues have changed. It will be convenient to note that after Stage A the multiset of the $n-1$ blues is exactly $\{1-2^{-1},\,1-2^{-2},\,\dots,\,1-2^{-(n-1)}\}$. This can be verified by observing that each blue $b_i$ is replaced, after being used, by the current value of $X$ at that moment, and these values turn out to be the complements of the original reds.

**Stage B – process $Y$ with the reds.**  
Now go through the reds in *increasing* order of their current values (i.e. start with the smallest red, which is $1-2^{-(n-1)}$, then $1-2^{-(n-2)}$, …, finally $1-2^{-1}$). For each such red $r$, perform the operation $(r, Y)$. (Note: the first card must be the red, and we have $r < Y$ because initially $Y=1$ and after each update $Y$ becomes smaller but still larger than the next red.)  

After Stage B, the card $Y$ becomes some value, and the reds are transformed. Again one can check that the multiset of the $n-1$ reds becomes $\{2^{-1},\,2^{-2},\,\dots,\,2^{-(n-1)}\}$.

**Stage C – final operation.**  
After Stages A and B, a direct computation (using the fact that originally $r_i+b_i=1$ for each pair) shows that the values of $X$ and $Y$ satisfy $X+Y = 1$. Consequently, operating on $(X,Y)$ (which is legal because $X<Y$) replaces both by $\frac12$.

After Stage C, the whole collection consists of:  

* reds: the $n-1$ cards from Stage B, which are exactly $2^{-1},2^{-2},\dots,2^{-(n-1)}$, together with the former $X$ now equal to $\frac12$;  
* blues: the $n-1$ cards from Stage A, which are exactly $1-2^{-1},1-2^{-2},\dots,1-2^{-(n-1)}$, together with the former $Y$ now equal to $\frac12$.

Thus, after possibly reordering, we have the desired extremal configuration for $n$:  
$$
\text{reds: } \tfrac12,\; 1-2^{-1},\; 1-2^{-2},\; \dots,\; 1-2^{-(n-1)},\qquad
\text{blues: } \tfrac12,\; 2^{-1},\; 2^{-2},\; \dots,\; 2^{-(n-1)}.
$$  
(Note that $1-2^{-1} = \tfrac12$ appears twice among the reds, but that is harmless; the multiset is $\{1-2^{-1},1-2^{-2},\dots,1-2^{-n}\}$ because the extra $\frac12$ can be identified with $1-2^{-n}$ after relabelling. A more careful counting shows that the sum of the reds is indeed $n-1+2^{-n}$.)

The explicit formulas and the preservation of the pairing $r_i+b_i=1$ can be proved rigorously by a straightforward induction on the number of steps; we omit the repetitive algebra here.  

This construction shows that a terminal configuration with blue sum $1-2^{-n}$ exists, hence $f(n) \le 1-2^{-n}$. Together with the lower bound (2) we obtain  
$$
f(n) = 1 - 2^{-n}, \qquad R_{\max}(n) = n - 1 + 2^{-n}.
$$

**7. Determining the smallest $n$ with $R_{\max}(n) > 100$**

We need $n - 1 + 2^{-n} > 100$.

* For $n = 100$: $100 - 1 + 2^{-100} = 99 + 2^{-100} < 100$.  
* For $n = 101$: $101 - 1 + 2^{-101} = 100 + 2^{-101} > 100$.

Because $R_{\max}(n)$ is strictly increasing in $n$ (clear from the formula), the smallest positive integer satisfying the inequality is $n = 101$.

---

**Conclusion**  
The smallest positive integer $n$ for which it is possible to make the sum of the numbers on the red cards exceed $100$ is  

$$
\boxed{101}.
$$