**Step 1. Reformulation.**  
Let $x_1,\dots,x_n \ge 0$ with $\sum_{i=1}^n x_i = 1$. Define  

$$
S_2 = \sum_{1\le i<j\le n} x_i x_j,\qquad
S_3 = \sum_{1\le i<j<k\le n} x_i x_j x_k.
$$

We first rewrite the two sums appearing in the inequality.  

From $\bigl(\sum x_i\bigr)^2 = 1$ we get  

$$
\sum_{i=1}^n x_i^2 = 1 - 2S_2. \tag{1}
$$

The sum of squared differences is  

$$
\begin{aligned}
\sum_{1\le i<j\le n} (x_i-x_j)^2
&= \sum_{i<j}(x_i^2 + x_j^2 - 2x_i x_j) \\
&= (n-1)\sum_{k=1}^n x_k^2 - 2S_2 \\
&= (n-1)(1-2S_2) - 2S_2 \\
&= n-1 - 2n S_2. \tag{2}
\end{aligned}
$$

Substituting (2) into the given inequality yields  

$$
S_2^2 \le \lambda S_3 + \frac{1}{n}\bigl(n-1 - 2n S_2\bigr)
= \lambda S_3 + \frac{n-1}{n} - 2S_2,
$$

which rearranges to  

$$
S_2^2 + 2S_2 - \frac{n-1}{n} \le \lambda S_3. \tag{3}
$$

**Step 2. The case $S_3 = 0$.**  
If at most two of the $x_i$ are positive, then $S_3 = 0$.  

- With exactly one positive variable (equal to $1$), we have $S_2 = 0$ and the left‑hand side of (3) equals $-\frac{n-1}{n} < 0$.  
- With exactly two positive variables $a, b$ where $a+b=1$, we have $S_2 = ab \le \frac14$ and  
  $$
  S_2^2 + 2S_2 \le \frac{1}{16} + \frac12 = \frac{9}{16} < \frac{n-1}{n}\quad (\text{since } \tfrac{n-1}{n} \ge \tfrac23 \text{ for } n\ge 3).
  $$  

Thus in both subcases the left‑hand side is negative, so (3) holds for any $\lambda$. Hence when proving (3) for all admissible vectors, only the situation $S_3 > 0$ is nontrivial.

**Step 3. Reformulation with deviations.**  
For $S_3 > 0$ we introduce the deviations from uniformity:  

$$
u_i = x_i - \frac{1}{n},\qquad i = 1,\dots,n.
$$  

Then $\sum_{i=1}^n u_i = 0$ and $u_i \ge -\frac{1}{n}$. Define  

$$
A = \sum_{i=1}^n u_i^2,\qquad B = \sum_{i=1}^n u_i^3.
$$

Now express $S_2$ and $S_3$ in terms of $A$ and $B$.  

First,  

$$
\sum_{i=1}^n x_i^2 = \sum_{i=1}^n \left(\frac{1}{n} + u_i\right)^2
= \frac{1}{n} + A,
$$  

so from (1),  

$$
S_2 = \frac{1}{2}\left(1 - \sum x_i^2\right)
= \frac{1}{2}\left(\frac{n-1}{n} - A\right)
= \frac{n-1}{2n} - \frac{A}{2}. \tag{4}
$$  

Next,  

$$
\sum_{i=1}^n x_i^3 = \sum_{i=1}^n \left(\frac{1}{n} + u_i\right)^3
= \frac{1}{n^2} + \frac{3}{n}A + B.
$$  

Using the identity $(\sum x_i)^3 = 1 = 3\sum x_i^2 - 2\sum x_i^3 + 6S_3$ (obtained by expanding $(\sum x_i)^3$), we get  

$$
S_3 = \frac{1}{6}\left(1 - 3\sum x_i^2 + 2\sum x_i^3\right)
= \frac{(n-1)(n-2)}{6n^2} - \frac{n-2}{2n}A + \frac{1}{3}B. \tag{5}
$$  

**Step 4. The key identity.**  
Define the candidate constant  

$$
c_n = \frac{3}{2}\cdot\frac{n-1}{n-2}.
$$  

Compute  

$$
S_2^2 = \left(\frac{n-1}{2n} - \frac{A}{2}\right)^2
= \frac{(n-1)^2}{4n^2} - \frac{n-1}{2n}A + \frac{A^2}{4}, \tag{6}
$$  

and  

$$
c_n S_3 = \frac{3}{2}\cdot\frac{n-1}{n-2}\left(\frac{(n-1)(n-2)}{6n^2} - \frac{n-2}{2n}A + \frac{1}{3}B\right)
= \frac{(n-1)^2}{4n^2} - \frac{3(n-1)}{4n}A + \frac{n-1}{2(n-2)}B. \tag{7}
$$  

Now form the difference  

$$
c_n S_3 - \left(S_2^2 + 2S_2 - \frac{n-1}{n}\right).
$$  

Using (4), (6) and (7),  

$$
\begin{aligned}
S_2^2 + 2S_2 - \frac{n-1}{n}
&= \left[\frac{(n-1)^2}{4n^2} - \frac{n-1}{2n}A + \frac{A^2}{4}\right] + \left[\frac{n-1}{n} - A\right] - \frac{n-1}{n} \\
&= \frac{(n-1)^2}{4n^2} - \frac{n-1}{2n}A + \frac{A^2}{4} - A.
\end{aligned}
$$  

Subtracting from (7),  

$$
\begin{aligned}
&c_n S_3 - \left(S_2^2 + 2S_2 - \frac{n-1}{n}\right) \\
&= \left[\frac{(n-1)^2}{4n^2} - \frac{3(n-1)}{4n}A + \frac{n-1}{2(n-2)}B\right]
   - \left[\frac{(n-1)^2}{4n^2} - \frac{n-1}{2n}A + \frac{A^2}{4} - A\right] \\
&= \frac{A}{4}\left(\frac{3n+1}{n} - A\right) + \frac{n-1}{2(n-2)}B. \tag{8}
\end{aligned}
$$  

**Step 5. A lower bound for $B$.**  
Since $x_i \ge 0$, we have $u_i = x_i - \frac{1}{n} \ge -\frac{1}{n}$. For $n \ge 3$, note that  

$$
\frac{n-2}{n-1} \ge \frac{1}{n} \quad \Longleftrightarrow\quad n^2 - 3n + 1 \ge 0,
$$  

which holds for $n \ge 3$. Therefore  

$$
u_i + \frac{n-2}{n-1} \ge 0.
$$  

Consequently,  

$$
u_i^3 + \frac{n-2}{n-1}u_i^2 = u_i^2\left(u_i + \frac{n-2}{n-1}\right) \ge 0.
$$  

Summing over $i = 1,\dots,n$ gives  

$$
B + \frac{n-2}{n-1}A \ge 0 \quad\Longrightarrow\quad B \ge -\frac{n-2}{n-1}A. \tag{9}
$$  

**Step 6. Non‑negativity of the key identity.**  
Substitute (9) into (8). Since $\frac{n-1}{2(n-2)} > 0$,  

$$
\begin{aligned}
c_n S_3 - \left(S_2^2 + 2S_2 - \frac{n-1}{n}\right)
&\ge \frac{A}{4}\left(\frac{3n+1}{n} - A\right) + \frac{n-1}{2(n-2)}\left(-\frac{n-2}{n-1}A\right) \\
&= \frac{A}{4}\left(\frac{3n+1}{n} - A - 2\right) \\
&= \frac{A}{4}\left(\frac{n+1}{n} - A\right). \tag{10}
\end{aligned}
$$  

Now we determine the range of $A$. From $x_i \ge 0$ and $\sum x_i = 1$, the maximum of $A$ occurs when one variable is $1$ and the rest are $0$, giving  

$$
A_{\max} = \left(1-\frac{1}{n}\right)^2 + (n-1)\left(-\frac{1}{n}\right)^2 = \frac{n-1}{n}.
$$  

Clearly $A \ge 0$. Thus $0 \le A \le \frac{n-1}{n}$. Moreover,  

$$
\frac{n+1}{n} - A \ge \frac{n+1}{n} - \frac{n-1}{n} = \frac{2}{n} > 0.
$$  

Hence the right‑hand side of (10) is non‑negative, and it vanishes only when $A = 0$, i.e. when all $u_i = 0$ (which means all $x_i = \frac{1}{n}$). Therefore, for every admissible vector we have  

$$
c_n S_3 \ge S_2^2 + 2S_2 - \frac{n-1}{n}, \tag{11}
$$  

with equality exactly when $x_1 = \dots = x_n = \frac{1}{n}$.

**Step 7. Optimality of $c_n$ for fixed $n$.**  
For the uniform distribution,  

$$
S_2 = \frac{n-1}{2n},\qquad S_3 = \frac{(n-1)(n-2)}{6n^2},
$$  

and a direct calculation shows  

$$
c_n S_3 = \frac{3}{2}\cdot\frac{n-1}{n-2} \cdot \frac{(n-1)(n-2)}{6n^2}
= \frac{(n-1)^2}{4n^2} = S_2^2 + 2S_2 - \frac{n-1}{n}.
$$  

Thus (11) is sharp: $c_n$ is the smallest constant such that (3) holds for **all** vectors with that particular $n$.

**Step 8. Behaviour of $c_n$ and the final condition on $\lambda$.**  
For $n \ge 3$,  

$$
c_n = \frac{3}{2}\left(1 + \frac{1}{n-2}\right) > \frac{3}{2},
$$  

and $c_n$ is strictly decreasing (since $\frac{1}{n-2}$ decreases) with  

$$
\lim_{n\to\infty} c_n = \frac{3}{2}.
$$  

* **Sufficiency of $\lambda > \frac{3}{2}$.**  
  Assume $\lambda > \frac{3}{2}$. Because $c_n \searrow \frac{3}{2}$, there exists an integer $N_0 \ge 3$ such that $c_{N_0} < \lambda$. Since the sequence is decreasing, $c_n \le c_{N_0} < \lambda$ for all $n \ge N_0$. For any $\mathbf{x}$ with $S_3 > 0$, inequality (11) gives  

  $$
  S_2^2 + 2S_2 - \frac{n-1}{n} \le c_n S_3 < \lambda S_3,
  $$  

  and for $S_3 = 0$ the left‑hand side is non‑positive (Step 2), so (3) holds as well. Therefore the original inequality is true for every $n \ge N_0$.

* **Necessity of $\lambda > \frac{3}{2}$.**  
  Suppose $\lambda \le \frac{3}{2}$ and that some integer $N$ has the required property. Take $n = \max\{N,3\}$ and the uniform vector $\mathbf{x} = (\frac{1}{n},\dots,\frac{1}{n})$. For this vector, Step 7 shows  

  $$
  S_2^2 + 2S_2 - \frac{n-1}{n} = c_n S_3 > \frac{3}{2} S_3 \ge \lambda S_3,
  $$  

  contradicting (3). Hence no such $N$ can exist.

**Conclusion**  
The inequality holds for all sufficiently large $n$ (with $N$ depending only on $\lambda$) if and only if $\lambda > \dfrac{3}{2}$.

$$
\boxed{\lambda > \dfrac{3}{2}}
$$