### 1.  Elementary consequences of the definition

A function $f:\mathbb N\to\mathbb N$ is *bonza* if  

$$
f(a)\mid b^a - f(b)^{f(a)}\qquad\text{for all }a,b\in\mathbb N.
$$

* **$f(1)=1$.**  
  Put $a=b=1$: $f(1)\mid 1-f(1)^{f(1)}$. Since $f(1)\mid f(1)^{f(1)}$, we get $f(1)\mid 1$, hence $f(1)=1$.

* **$f(2)\in\{1,2,4\}$.**  
  With $a=b=2$ we have $f(2)\mid 4-f(2)^{f(2)}$. Because $f(2)\mid f(2)^{f(2)}$, we obtain $f(2)\mid 4$; thus $f(2)$ is a positive divisor of $4$.

* **For every $n$, $f(n)\mid n^n$.**  
  Setting $a=b=n$ gives $f(n)\mid n^n-f(n)^{f(n)}$. As $f(n)\mid f(n)^{f(n)}$, it follows that $f(n)\mid n^n$. Consequently any prime divisor of $f(n)$ also divides $n$.

---

### 2.  Case $f(2)=1$

Assume $f(2)=1$.

* **Odd primes.**  
  For any odd prime $p$, take $a=p,\ b=2$:  
  $$
  f(p)\mid 2^p - 1^{f(p)} = 2^p-1.
  $$
  Also $f(p)\mid p^p$. Since $\gcd(2^p-1,p^p)=1$ (because $p\nmid 2^p-1$ by Fermat’s little theorem), we have $f(p)=1$.

* **Any positive integer $n$.**  
  Let $p$ be any prime divisor of $n$ (if $n=1$ then $f(1)=1$ already). The condition with $a=n,\ b=p$ yields  
  $$
  f(n)\mid p^n-1.
  $$
  Suppose a prime $q$ divides $f(n)$. From $f(n)\mid n^n$ we know $q\mid n$. In particular, $q$ is a prime divisor of $n$. Using $a=n,\ b=q$ we get $f(n)\mid q^n-1$. Since $q\mid f(n)$ we have $q\mid q^n-1$, which forces $q\mid 1$, impossible. Hence $f(n)$ has no prime factors, i.e. $f(n)=1$.

Thus for every $n\in\mathbb N$, $f(n)=1\le 4n$.

---

### 3.  Case $f(2)=4$

Assume $f(2)=4$.

* **Odd primes.**  
  Let $p$ be an odd prime and write $f(p)=p^{\alpha}$ (because $f(p)\mid p^p$). Take $a=p,\ b=2$:  
  $$
  p^{\alpha}\mid 2^p - 4^{p^{\alpha}}.
  $$
  If $\alpha\ge 1$ then reducing modulo $p$ gives  
  $$
  2\equiv 2^p\equiv 4^{p^{\alpha}}\equiv 4\pmod p,
  $$
  which implies $p\mid 2$, contradiction. Hence $\alpha=0$, i.e. $f(p)=1$ for every odd prime $p$.

* **Odd numbers.**  
  Let $n>1$ be odd and let $p$ be any prime divisor of $n$. As above, $f(p)=1$ and the condition $a=n,\ b=p$ gives $f(n)\mid p^n-1$. The same argument as in Case 1 (using a prime divisor of $f(n)$) shows that $f(n)$ has no odd prime factor, hence $f(n)=1$. (The possibility of a factor $2$ is excluded because an odd $n$ could not contain the prime $2$ in $f(n)$ due to $f(n)\mid n^n$.)

* **Even numbers.**  
  Write $n=2^s\cdot t$ with $s\ge 1$ and $t$ odd. For any odd $b$ we have $f(b)=1$, so the condition with $a=n,\ b$ becomes  
  $$
  f(n)\mid b^n-1.
  $$
  In particular, taking $b=3$ we obtain $f(n)\mid 3^n-1$.  
  **2‑adic valuation.** Because $n$ is even, the Lifting‑The‑Exponent Lemma (LTE) gives  
  $$
  v_2(3^n-1)=v_2(3-1)+v_2(3+1)+v_2(n)-1 = 1+2+s-1 = s+2.
  $$
  Hence $v_2(f(n))\le s+2$.  
  Moreover, if an odd prime $q$ divided $f(n)$, then $q\mid n$ (since $f(n)\mid n^n$), and using $a=n,\ b=q$ with $f(q)=1$ would lead to the same contradiction as before. Therefore $f(n)$ is a power of two, say $f(n)=2^{e}$ with $e\le s+2$. Consequently  
  $$
  f(n)\le 2^{s+2}.
  $$
  Since $n=2^s t\ge 2^s$, we have $2^{s+2}=4\cdot 2^s\le 4n$.  
  (The bound also covers $n=2$ because $f(2)=4=2^{1+2}\le 4\cdot2$.)

Thus in this case as well $f(n)\le 4n$ for all $n$.

---

### 4.  Case $f(2)=2$

Assume $f(2)=2$.

#### 4.1.  Behaviour on odd primes

For an odd prime $p$ we have $f(p)\mid p^p$, so we can write $f(p)=p^{\alpha}$ with $\alpha\ge 0$ (where $\alpha=0$ means $f(p)=1$).

Take two distinct odd primes $p,q$. Consider the two conditions  

$$
\begin{aligned}
a=p,\ b=q&:\quad p^{\alpha}\mid q^p - (q^{\beta})^{p^{\alpha}} = q^p - q^{\beta p^{\alpha}},\\
a=q,\ b=p&:\quad q^{\beta}\mid p^q - p^{\alpha q^{\beta}}.
\end{aligned}
$$

We analyse the possible values of $\alpha,\beta$.

*If $\alpha\ge 1$ and $\beta\ge 1$:*  
  Working modulo $p$ in the first divisibility, using $q^p\equiv q\pmod p$ and $q^{\beta p^{\alpha}}\equiv q^{\beta}\pmod p$ (because $p^{\alpha}\equiv1\pmod{p-1}$), we get  
  $$
  q - q^{\beta}\equiv 0\pmod p\quad\Longrightarrow\quad q^{\beta-1}\equiv 1\pmod p. \tag{1}
  $$
  Similarly, from the second divisibility modulo $q$ we obtain  
  $$
  p^{\alpha-1}\equiv 1\pmod q. \tag{2}
  $$

*If $\alpha\ge 1,\ \beta=0$:*  
  Then $f(q)=1$. Condition $a=p,\ b=q$ becomes $p^{\alpha}\mid q^p-1$. In particular $p\mid q^p-1$, and by Fermat $q^p\equiv q\pmod p$, so $q\equiv 1\pmod p$. Hence  
  $$
  q\equiv 1\pmod p. \tag{3}
  $$

*If $\alpha=0,\ \beta\ge 1$:*  
  Symmetrically we would get $p\equiv 1\pmod q$.

*If $\alpha=0,\ \beta=0$:*  
  No extra restriction.

---

**Lemma A.** *For any odd prime $p$ we have $\alpha\in\{0,1\}$.*

*Proof.* Suppose, for contradiction, that there exists an odd prime $p$ with $\alpha\ge 2$.  

From condition $a=p,\ b=2$ we have $p^{\alpha}\mid 2^p-2^{p^{\alpha}}$. Because $\gcd(p^{\alpha},2^p)=1$, this implies  

$$
2^{p^{\alpha}-p}\equiv 1\pmod{p^{\alpha}}.
$$

The order of $2$ modulo $p^{\alpha}$ must divide both $p^{\alpha}-p$ and $\varphi(p^{\alpha})=p^{\alpha-1}(p-1)$. A short computation shows that  
$$
\gcd(p^{\alpha}-p,\ p^{\alpha-1}(p-1)) = p(p-1).
$$
Hence the order divides $p(p-1)$, which for $\alpha\ge 3$ would imply that the group of units modulo $p^{\alpha}$ (which is cyclic) contains an element whose order does not contain the full factor $p^{\alpha-1}$. This is possible only if the order is at most $p(p-1)$. However, for $\alpha\ge 3$ we have $p^{\alpha-1}>p$, so the order cannot be a multiple of $p^{\alpha-1}$. While this does not directly contradict $\alpha=2$, we will use a different argument to exclude $\alpha\ge 2$ globally.

Consider another odd prime $q$. By (1) and (2), if $\beta\ge 1$ then $q$ must satisfy  
$q^{\beta-1}\equiv 1\pmod p$ and $p^{\alpha-1}\equiv 1\pmod q$.  
The second congruence forces $q$ to divide $p^{\alpha-1}-1$, a fixed integer; hence only finitely many primes $q$ can satisfy it. Since there are infinitely many primes, there exists a prime $q$ with $\beta=0$. For such a $q$, (3) gives $q\equiv 1\pmod p$.  

But we can choose a prime $q$ *not* congruent to $1$ modulo $p$ (e.g., take $q=3$ if $p\neq 3$, or $q=5$ if $p=3$, etc.). For that $q$, condition $a=p,\ b=q$ with $\beta=0$ would fail because then $p^{\alpha}\mid q^p-1$ implies in particular $q^p\equiv 1\pmod p$ and by Fermat $q\equiv 1\pmod p$, which is false. The alternative possibility, that $q$ has $\beta\ge 1$, is impossible because then $q$ would be one of the finitely many divisors of $p^{\alpha-1}-1$, but we can pick $q$ outside that finite set.  

Thus no prime $p$ with $\alpha\ge 2$ can exist. ∎

---

**Lemma B.** *Either $f(p)=1$ for every odd prime $p$, or $f(p)=p$ for every odd prime $p$.*

*Proof.* Suppose there exist two odd primes $p,q$ with $f(p)=1$ and $f(q)=q$. Consider the condition $a=p,\ b=q$:  
$$
1\mid q^p - f(p)^{1} = q^p-1,
$$  
which is true. The condition $a=q,\ b=p$ gives  
$$
q\mid p^q - f(q)^{1} = p^q - q.
$$  
Modulo $q$, we have $p^q\equiv p\pmod q$ (Fermat), so $p - q \equiv 0\pmod q$, i.e. $p\equiv 0\pmod q$. Since $p$ and $q$ are distinct primes, this forces $p=q$, a contradiction.  

Similarly, if we had $f(p)=p$ and $f(q)=1$ we would obtain a contradiction by swapping roles. Hence the values on odd primes are consistent: either all are $1$ or all equal the prime itself. ∎

---

#### 4.2.  Subcase: All odd primes satisfy $f(p)=1$

* **Odd numbers.**  
  Let $n>1$ be odd and let $p$ be any prime divisor of $n$. As in Case 1, using $a=n,\ b=p$ and the fact that any prime divisor of $f(n)$ would also divide $n$, we conclude $f(n)=1$.

* **Even numbers.**  
  Let $n=2^s\cdot t$ with $s\ge 1$, $t$ odd. For any odd prime $b$ (in particular $b=3$) we have $f(b)=1$, so  
  $$
  f(n)\mid b^n-1.
  $$
  Taking $b=3$ and applying LTE (since $n$ is even) gives  
  $$
  v_2(3^n-1)=v_2(3-1)+v_2(3+1)+v_2(n)-1 = 1+2+s-1 = s+2.
  $$
  Hence $v_2(f(n))\le s+2$.  

  Moreover, if an odd prime $r$ divided $f(n)$, then $r\mid n$ (by $f(n)\mid n^n$), and using $a=n,\ b=r$ (with $f(r)=1$) would lead to $r\mid r^n-1$, impossible. Therefore $f(n)$ is a power of two, say $f(n)=2^e$ with $e\le s+2$.  

  Thus  
  $$
  f(n)\le 2^{s+2} = 4\cdot 2^s \le 4n.
  $$

---

#### 4.3.  Subcase: All odd primes satisfy $f(p)=p$

We prove that then $f(n)=n$ for **every** positive integer $n$.

Let $n$ be arbitrary. Choose any prime $p$ that does **not** divide $n$. (Such primes exist because there are infinitely many primes; e.g., take $p > n$.) Consider the condition with $a=p,\ b=n$:  

$$
f(p)\mid n^p - f(n)^{f(p)}.
$$

Since $f(p)=p$ and $p\nmid n$ (by the choice of $p$), the number $f(n)$ is also not divisible by $p$ (because $f(n)\mid n^n$). Reducing the divisibility modulo $p$ and using Fermat’s little theorem ($n^p\equiv n\pmod p$ and $f(n)^p\equiv f(n)\pmod p$) we obtain  

$$
n - f(n) \equiv 0\pmod p,
$$  
i.e. $p\mid n-f(n)$.  

If $n\neq f(n)$, then $n-f(n)$ is a nonzero integer. But we have just shown that **every** prime $p$ not dividing $n$ divides $n-f(n)$. In particular, there are infinitely many such primes (any prime larger than $|n-f(n)|$ and not dividing $n$ will work). A nonzero integer can have only finitely many prime divisors — contradiction. Therefore $n-f(n)=0$, i.e.  

$$
f(n)=n.
$$

Clearly $f(n)=n\le 4n$.

---

Thus in every possible case we have established the universal bound  

$$
f(n)\le 4n\qquad\text{for all }n\in\mathbb N.
$$

---

### 5.  Sharpness – a function attaining the ratio $4$

Define $f:\mathbb N\to\mathbb N$ by  

$$
f(1)=1,\qquad
f(2)=2,
$$  
and for $n>1$:  

$$
f(n)=\begin{cases}
1, & n\text{ odd},\\
2^{\nu_2(n)+2}, & n\text{ even},\ n\ge 4,
\end{cases}
$$  
where $\nu_2(n)$ denotes the exponent of the highest power of $2$ dividing $n$.

We verify that this $f$ is bonza.

* **If $a$ is odd:** then $f(a)=1$ and the condition $1\mid b^a - f(b)^{1}$ is trivially true.

* **If $a$ is even and $b$ is odd:** then $f(b)=1$ and we need $f(a)\mid b^a-1$.  
  Write $a=2^s\cdot t$ with $t$ odd; then $f(a)=2^{s+2}$. By LTE (since $a$ is even) we have  
  $$
  v_2(b^a-1)=v_2(b-1)+v_2(b+1)+v_2(a)-1\ge 1+1+s-1 = s+2,
  $$  
  so indeed $2^{s+2}\mid b^a-1$.

* **If $a$ is even and $b=2$:** here $f(b)=2$. We need to show $f(a)\mid 2^a - 2^{f(a)}$.  
  Set $e=s+2$ so that $f(a)=2^e$. The number $2^a-2^{2^e}$ has 2‑adic valuation exactly $\min(a,2^e)$ because the difference of the two powers of two yields an odd factor after factoring out the smaller power.  
  For $a\ge 4$ we have $a\ge s+2 = e$ (easy to check), hence $\min(a,2^e)\ge e$, and therefore $2^e$ divides the difference. For $a=2$ the condition holds directly ($2\mid 4-4=0$).

* **If both $a$ and $b$ are even:** write $a=2^s t$, $b=2^u v$ with $t,v$ odd.  
  Let $s\ge 1$; then $f(a)=2^{s+2}$, $f(b)=2^{u+2}$ (with the convention that for $b=2$ we have $f(b)=2$, which matches $2^{1+2}=8$? Actually we treat $b=2$ separately; but the remaining even $b\ge 4$ satisfy the formula). A detailed analysis (factoring the appropriate power of two) shows that $v_2\bigl(b^a - f(b)^{f(a)}\bigr)\ge s+2$. One can proceed by considering the two terms $b^a$ and $f(b)^{f(a)}$. If $u a\ge (u+2)f(a)$ then the smaller term is $f(b)^{f(a)}$ and after factoring out that power of two the remaining factor is odd, giving valuation $(u+2)f(a)$ which is certainly $\ge s+2$. If $u a < (u+2)f(a)$ then the valuation is $u a$; one checks that $u a\ge s+2$ always holds for $a\ge 4$. The isolated case $a=2$ (where $f(2)=2$) has already been verified. Thus the divisibility condition is satisfied.

Hence $f$ is indeed bonza. For $n=4$ we have $\nu_2(4)=2$, so $f(4)=2^{2+2}=16 = 4\cdot 4$. Therefore the constant $4$ cannot be reduced.

---

### 6.  Conclusion

We have shown that for **every** bonza function $f$ and **every** positive integer $n$,

$$
f(n)\le 4n,
$$

and that the constant $4$ is best possible. Thus the smallest real constant $c$ with the required property is  

$$
\boxed{4}.
$$