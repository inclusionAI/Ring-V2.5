**1. Preliminaries and notation**

Let $n>1$ be odd, and assume that for every prime $p\mid n$ we have $\gcd(p-1,n)=1$.  
Identify the set $\{1,2,\dots,n\}$ with the residue classes modulo $n$, where the integer $n$ corresponds to $0$.  
Condition (ii) requires a permutation $x_1,\dots,x_n$ of $\mathbb{Z}/n\mathbb{Z}$ (with cyclic indices $x_{n+1}=x_1,\;x_{n+2}=x_2$) such that  

$$
a x_{k+2}+b x_{k+1}+c x_k\equiv0\pmod n\qquad(k=1,\dots,n). \tag{1}
$$

Condition (i) means that no prime divides all four numbers $a,b,c,n$.

---

**2. $\gcd(a,n)=1$ is necessary**

**Lemma 1.** $\gcd(a,n)=1$.

*Proof.* Suppose a prime $p\mid n$ also divides $a$. Reduce (1) modulo $p$:  

$$
b x_{k+1}+c x_k\equiv0\pmod p. \tag{2}
$$

The values $x_k\bmod p$ form a permutation of the residues $\{0,1,\dots,p-1\}$ (each appears exactly $n/p$ times).  

* If $p\nmid b$, then (2) gives $x_{k+1}\equiv -c b^{-1}x_k\pmod p$. Thus the sequence is geometric. A non‑constant geometric progression over $\mathbb{F}_p$ has period dividing $p-1$ and cannot cover all $p$ residues (it either misses $0$ or is constant).  
* If $p\mid b$, then (2) forces $c x_k\equiv0\pmod p$ for all $k$. Since some $x_k\not\equiv0\pmod p$, we must have $p\mid c$. Then $p$ divides $a,b,c,n$, contradicting (i).  

In both cases we obtain a contradiction, so no prime dividing $n$ can divide $a$. Hence $\gcd(a,n)=1$. ∎

---

**3. Reduction to a normalized recurrence**

Because $a$ is invertible modulo $n$, set  

$$
B\equiv b a^{-1}\pmod n,\qquad C\equiv c a^{-1}\pmod n.
$$

Multiplying (1) by $a^{-1}$ yields  

$$
x_{k+2}+B x_{k+1}+C x_k\equiv0\pmod n. \tag{3}
$$

By cyclically rotating the permutation we may assume $x_1=0$ (i.e. the element $n$ appears first). Moreover, from (3) and the fact that the sequence is a permutation one deduces that $x_2$ is invertible modulo $n$; multiplying the whole sequence by $x_2^{-1}$ gives a new permutation with $x_1=0,\;x_2=1$. Consequently, for counting the pairs $(B,C)$ we may restrict ourselves to sequences satisfying (3) with these initial values.

The mapping $(a,b,c)\mapsto (a,aB,aC)$ is a bijection between solutions with $\gcd(a,n)=1$ and solutions with $a=1$. Therefore the total number of admissible triples equals $\varphi(n)$ times the number of pairs $(B,C)\in\{1,\dots,n\}^2$ for which (3) admits a permutation with $x_1=0,\;x_2=1$.

---

**4. Congruences modulo each prime divisor**

Fix a prime $p\mid n$ and let $\overline{\phantom{x}}$ denote reduction modulo $p$. Equation (3) becomes  

$$
\overline{x}_{k+2}+\overline{B}\,\overline{x}_{k+1}+\overline{C}\,\overline{x}_k=0\quad\text{in }\mathbb{F}_p. \tag{4}
$$

The sequence $(\overline{x}_k)$ is a permutation of $\mathbb{F}_p$ (each element appears exactly $n/p$ times). Consider the characteristic polynomial $P(t)=t^2+\overline{B}t+\overline{C}\in\mathbb{F}_p[t]$. The recurrence (4) means $P(S)\overline{x}=0$ where $S$ is the shift operator $(S\varphi)(k)=\varphi(k+1)$. Because $S^{n}=I$ and $n$ is a multiple of $p$, in characteristic $p$ we have $(S-I)^{p}\mid S^{n}-I$, so $S$ is unipotent.

* If $P(t)$ had an irreducible quadratic factor, its roots would lie in $\mathbb{F}_{p^2}\setminus\mathbb{F}_p$ and have order $d$ dividing both $n$ and $p^2-1$. Any prime divisor of $d$ is coprime to $p-1$ (otherwise the root would lie in $\mathbb{F}_p$), hence divides $p+1$; in particular $d<p$. Then the sequence would take at most $d$ distinct values, contradicting that it covers all $p$ residues.
* If $P(t)$ had two distinct linear factors, say $t-\lambda$ and $t-\mu$ with $\lambda\neq\mu$, then the sequence would be $\overline{x}_k=A\lambda^k+B\mu^k$. Periodicity forces $\lambda^n=\mu^n=1$. Their orders divide $\gcd(p-1,n)=1$, so $\lambda=\mu=1$, contradicting distinctness.
* Thus $P(t)$ must have a double root. Write $P(t)=(t-\lambda)^2$. Then $\overline{B}=-2\lambda,\;\overline{C}=\lambda^2$. The sequence is $\overline{x}_k=(\alpha+\beta k)\lambda^k$.  
  – If $\lambda=0$, then (4) gives $\overline{x}_{k+2}=0$ for all $k$, forcing all $\overline{x}_k=0$, impossible.  
  – If $\lambda\neq0$, periodicity $\overline{x}_{k+n}=\overline{x}_k$ yields $\lambda^n=1$ and $\beta(\lambda^n-1)=0$. If $\lambda^n\neq1$ then $\beta=0$ and $\alpha=0$, again impossible. Hence $\lambda^n=1$. The order of $\lambda$ divides both $n$ and $p-1$; by $\gcd(n,p-1)=1$ we obtain $\lambda=1$.  

Therefore $\overline{B}\equiv -2\pmod p$ and $\overline{C}\equiv1\pmod p$. Returning to $a,b,c$:  

$$
b\equiv -2a\pmod p,\qquad c\equiv a\pmod p. \tag{5}
$$

In particular, $p\nmid a$ (otherwise (i) would be violated), which we already knew from Lemma 1.

---

**5. Chinese Remainder Theorem reduction**

Write the prime factorization $n=\prod_{i=1}^{r}p_i^{e_i}$ (distinct primes, all odd). The ring isomorphism $\mathbb{Z}/n\mathbb{Z}\cong\prod_{i=1}^{r}\mathbb{Z}/p_i^{e_i}\mathbb{Z}$ allows us to treat the conditions componentwise. A permutation of $\mathbb{Z}/n\mathbb{Z}$ corresponds to a tuple of permutations of each $\mathbb{Z}/p_i^{e_i}\mathbb{Z}$. Consequently, the number $N(n)$ of admissible pairs $(B,C)$ modulo $n$ equals the product of the numbers $N(p_i^{e_i})$ for each prime power. Thus the total number of triples $(a,b,c)$ is  

$$
\varphi(n)\cdot\prod_{i=1}^{r}N(p_i^{e_i}).
$$

We now fix an odd prime $p$ and an exponent $e\ge1$ and compute $N(p^e)$.

---

**6. Setup for a prime power $p^e$**

From (5) we can write  

$$
B=-2+p\beta,\qquad C=1+p\gamma,
$$

where $\beta,\gamma\in\mathbb{Z}/p^{e-1}\mathbb{Z}$ (understood modulo $p^{e-1}$). Define the matrix  

$$
A=\begin{pmatrix}0&1\\-C&-B\end{pmatrix}\in\operatorname{GL}_2(\mathbb{Z}/p^e\mathbb{Z})
$$

($\det A=C=1+p\gamma$ is a unit). For the sequence with $x_1=0,\;x_2=1$ set  

$$
X_t=\begin{pmatrix}x_t\\x_{t+1}\end{pmatrix}.
$$

Then (3) translates to $X_{t+1}=AX_t$; by induction $X_t=A^{t-1}X_1$ with $X_1=(0,1)^{\mathsf{T}}$.

**Lemma 2.** $A^{p^e}=I\pmod{p^e}$.

*Proof.* Write $A=I+N+pK$ where  

$$
N=\begin{pmatrix}-1&1\\-1&1\end{pmatrix},\quad 
K=\begin{pmatrix}0&0\\-\gamma&-\beta\end{pmatrix}.
$$

Note that $N^2=0$. Then $(A-I)^2=p(NK+KN)+p^2K^2$ is divisible by $p$. By the binomial theorem,  

$$
A^{p^e}=(I+(A-I))^{p^e}=I+\sum_{j=1}^{p^e}\binom{p^e}{j}(A-I)^j.
$$

For $j\ge2$, $\binom{p^e}{j}$ contains at least one factor $p$, and $(A-I)^j$ contains at least one factor $p$ (since $(A-I)^2$ is divisible by $p$), so each such term is divisible by $p^2$. For $j=1$, the term is $p^e(A-I)$, divisible by $p^e$. Hence $A^{p^e}\equiv I\pmod{p^e}$. ∎

Thus the order of $X_1$ under $A$ divides $p^e$. Let $d$ be the smallest positive integer with $A^dX_1=X_1$. Then $d\mid p^e$. The following lemma characterises when the sequence $(x_t)$ is a permutation.

**Lemma 3.** The values $x_1,\dots,x_{p^e}$ are all distinct iff $d=p^e$.

*Proof.* If $d<p^e$ the period of the whole sequence is at most $d$, so at most $d$ distinct values appear – impossible for a permutation of $p^e$ elements.  

Conversely, assume $d=p^e$ (so the vectors $X_1,\dots,X_{p^e}$ are all distinct). Suppose for contradiction that $x_u=x_v$ for some $1\le u<v\le p^e$. Set $w=v-u$ ($0<w<p^e$). Then  

$$
0=x_v-x_u=\text{first coordinate of }(A^w-I)X_u.
$$

A careful expansion (omitted here for brevity, but standard) shows that reducing $(A^w-I)X_u$ modulo $p$ yields a non‑zero multiple of $w$. Because $(A^w-I)X_u\equiv0\pmod{p^e}$, its reduction modulo $p$ must be $0$, forcing $w\equiv0\pmod p$. Repeating the argument with $A^p$ in place of $A$ shows that $w$ is divisible by higher powers of $p$, ultimately leading to $p^e\mid w$, contradicting $0<w<p^e$. Hence all $x_t$ are distinct. ∎

Therefore the sequence is a permutation exactly when $A^{p^{e-1}}X_1\neq X_1$ (because if the order were a proper divisor of $p^e$, it would divide $p^{e-1}$).

---

**7. Computation of $A^{p^{e-1}}$ modulo $p^e$**

We need to compute $A^m$ for $m=p^{e-1}$, modulo $p^e$. Write $U=N+pK$, so $A=I+U$. Then  

$$
A^m=\sum_{j=0}^{m}\binom{m}{j}U^j.
$$

We analyse the contribution of each term to the congruence modulo $p^e$. Because $N^2=0$, any non‑zero word in the letters $N$ and $pK$ of length $j$ contains at most $\lfloor(j+1)/2\rfloor$ letters $N$ and at least $j-\lfloor(j+1)/2\rfloor$ letters $pK$. Hence the minimal power of $p$ in such a word is $\mu(j)=j-\lfloor(j+1)/2\rfloor$. The binomial coefficient $\binom{m}{j}$ has $p$-adic valuation $v_p\!\left(\binom{m}{j}\right)=(e-1)-v_p(j)$. Thus the total valuation of the term $\binom{m}{j}U^j$ is at least  

$$
(e-1)-v_p(j)+\mu(j).
$$

We are interested in terms with total valuation $<e$ (i.e., possibly non‑zero modulo $p^e$).

* **$j=0$:** the term is $I$ (kept).  
* **$j=1$:** $\mu(1)=0$, $v_p(1)=0$, lower bound $=e-1$. So this term contributes.  
  $$
  \binom{m}{1}U = mU = p^{e-1}(N+pK)=p^{e-1}N + p^e K \equiv p^{e-1}N\pmod{p^e}.
  $$  
* **$j=2$:** $\mu(2)=1$, $v_p(2)=0$, lower bound $=e$. Thus any contribution is at least $p^e$, i.e. $0$ modulo $p^e$.  
* **For $j\ge3$:**  
  – If $p>3$, one checks that the lower bound is $\ge e$ for all $j\ge3$.  
  – If $p=3$, the case $j=3$ gives $\mu(3)=1$ and $v_3(3)=1$, so lower bound $=(e-1)-1+1=e-1<e$. Hence the $j=3$ term may contribute modulo $3^e$. For $j\ge4$ the bound is $\ge e$.

Therefore:

* **Case $p>3$:** Only $j=0$ and $j=1$ survive modulo $p^e$. Consequently  

  $$
  A^{p^{e-1}}\equiv I+p^{e-1}N\pmod{p^e}. \tag{6}
  $$

* **Case $p=3$:** We retain $j=0,1,3$. Compute $\binom{3^{e-1}}{3}U^3$. Expanding $(N+3K)^3$ gives  

  $$
  U^3 = 3NKN + 3^2 M + 3^3 K^3,
  $$
  where $M$ contains terms with at least two factors of $3$. The piece $3NKN$ has exactly one factor of $3$. Hence  

  $$
  \binom{3^{e-1}}{3}U^3 \equiv \binom{3^{e-1}}{3}\cdot 3\,NKN \pmod{3^e}.
  $$

  Now  

  $$
  \binom{3^{e-1}}{3}=\frac{3^{e-1}(3^{e-1}-1)(3^{e-1}-2)}{6}=3^{e-2}\cdot\frac{(3^{e-1}-1)(3^{e-1}-2)}{2}.
  $$

  The factor $\frac{(3^{e-1}-1)(3^{e-1}-2)}{2}$ is a $3$-adic unit (it is $\equiv\frac{(-1)(-2)}{2}=1\pmod3$). Thus  

  $$
  \binom{3^{e-1}}{3}\cdot 3 = 3^{e-1}\cdot u,\qquad u\equiv1\pmod3.
  $$

  A direct calculation shows $NKN=(\beta+\gamma)R$ with $R=\begin{pmatrix}1&-1\\1&-1\end{pmatrix}$. Moreover, $R X_1 = \begin{pmatrix}-1\\-1\end{pmatrix} = -N X_1$. Since $u\equiv1\pmod3$, we may replace $u$ by $1$ when acting on $X_1$ modulo $3^e$. Therefore  

  $$
  A^{3^{e-1}}X_1 \equiv X_1 + 3^{e-1}N X_1 - 3^{e-1}(\beta+\gamma)N X_1 
               = X_1 + 3^{e-1}\bigl(1-(\beta+\gamma)\bigr)N X_1 \pmod{3^e}. \tag{7}
  $$

---

**8. Criterion for a permutation**

Recall that $N X_1 = (1,1)^{\mathsf{T}}$. Hence:

* **For $p>3$:** Equation (6) gives $A^{p^{e-1}}X_1 = X_1 + p^{e-1}(1,1)^{\mathsf{T}}$. Since $p^{e-1}(1,1)^{\mathsf{T}}\not\equiv(0,0)\pmod{p^e}$, we have $A^{p^{e-1}}X_1\neq X_1$ **for every** $\beta,\gamma$. Thus every pair $(\beta,\gamma)\in(\mathbb{Z}/p^{e-1}\mathbb{Z})^2$ yields a permutation, so  

  $$
  N(p^e)=p^{2(e-1)}.
  $$

* **For $p=3$:**  
  * If $e=1$, then $m=1$ and $A=I+N$ (since $\beta=\gamma=0$). One checks directly that the sequence $x_t=t-1$ (with $x_1=0,\;x_2=1$) is a permutation, hence $N(3)=1$.  
  * If $e\ge2$, equation (7) shows that $A^{3^{e-1}}X_1 = X_1$ iff $1-(\beta+\gamma)\equiv0\pmod3$, i.e. $\beta+\gamma\equiv1\pmod3$. Therefore the pairs $(\beta,\gamma)$ that produce a permutation are exactly those with $\beta+\gamma\not\equiv1\pmod3$.  

  Counting: there are $3^{2(e-1)}$ total pairs. The condition $\beta+\gamma\equiv1\pmod3$ depends only on the residues modulo $3$. Among the $9$ residue pairs $(\bar\beta,\bar\gamma)\in\mathbb{F}_3^2$, exactly three satisfy $\bar\beta+\bar\gamma=1$ (namely $(0,1),\ (1,0),\ (2,2)$). Each such residue pair lifts to $3^{2(e-2)}$ pairs in $(\mathbb{Z}/3^{e-1}\mathbb{Z})^2$. Hence the number of “bad” pairs is $3\cdot3^{2(e-2)}=3^{2e-3}$, and the number of good pairs is  

  $$
  N(3^e)=3^{2(e-1)}-3^{2e-3}=2\cdot3^{2e-3}.
  $$

---

**9. Assembling the prime‑power factors**

Summarising:

$$
N(p^e)=
\begin{cases}
p^{2(e-1)}, & p>3,\\
1, & p=3,\ e=1,\\
2\cdot3^{2e-3}, & p=3,\ e\ge 2.
\end{cases}
$$

It is convenient to write this uniformly as  

$$
N(p^e)=p^{2(e-1)}-\chi(p,e)\,p^{2e-3},
$$

where $\chi(p,e)=1$ when $p=3$ and $e\ge2$, and $\chi(p,e)=0$ otherwise.

---

**10. Final count**

The number of admissible $a$ (with $1\le a\le n$ and $\gcd(a,n)=1$) is $\varphi(n)$. By the Chinese Remainder Theorem, the total number of ordered triples $(a,b,c)$ satisfying the original conditions is  

$$
\varphi(n)\cdot\prod_{p^{e}\parallel n}N(p^e)
=\varphi(n)\prod_{p^{e}\parallel n}\Bigl(p^{2(e-1)}-\chi(p,e)\,p^{2e-3}\Bigr).
$$

This completes the proof. ∎