**Preliminaries.**  
A line is called *sunny* if its slope is not $0$, $\infty$ or $-1$, i.e. it is not parallel to the $x$-axis, the $y$-axis, or the line $x+y=0$.  
For $n\ge 3$ we consider the finite set  
$$
S_n = \{(a,b)\in \mathbb{N}^2 \mid a\ge 1,\;b\ge 1,\;a+b\le n+1\}.
$$  
A covering of $S_n$ by $n$ distinct lines is required.

---

### 1.  Constructions (sufficiency)

* **$k=0$.**  
  Take the $n$ vertical lines $x=1,x=2,\dots,x=n$. Every point $(a,b)\in S_n$ satisfies $1\le a\le n$, hence lies on the line $x=a$.

* **$k=1$.**  
  Use the $n-1$ vertical lines $x=1,\dots,x=n-1$ and one sunny line $\ell$ through $(n,1)$ with slope $1$, for instance $\ell:\;y=x-n+1$. The vertical lines cover all points with $a\le n-1$; the point $(n,1)$ lies on $\ell$.

* **$k=3$.**  

  *For $n=3$:* One can take  
  $$
  L_1:\;y=x,\qquad
  L_2:\;\text{through }(1,2)\text{ and }(3,1),\qquad
  L_3:\;\text{through }(1,3)\text{ and }(2,1).
  $$  
  A direct check shows that the six points of $S_3$ are covered and exactly these three lines are sunny (their slopes are $1,-1/2,-2$).

  *For $n\ge 4$:* Set $m=n-3$. Choose the $m$ vertical lines $x=1,\dots,x=m$.  
  Additionally, take three sunny lines  
  $$
  \begin{aligned}
  L_1&:\;\text{through }(m+1,1)\text{ and }(m+2,2),\\
  L_2&:\;\text{through }(m+1,2)\text{ and }(m+3,1),\\
  L_3&:\;\text{through }(m+1,3)\text{ and }(m+2,1).
  \end{aligned}
  $$  
  Their slopes are $1,\,-\frac12,\,-2$ (all sunny).  
  Points with $x\le m$ are covered by the vertical lines.  
  The remaining points of $S_n$ are exactly the six points with $x=m+1,m+2,m+3$ and $y$ satisfying $x+y\le n+1$; they lie on $L_1,L_2,L_3$ in pairs as listed. Hence all $n$ lines together cover $S_n$.

Thus every $k\in\{0,1,3\}$ is attainable.

---

### 2.  A lemma for $n\ge 4$

Define the three *special* lines  
$$
A_1:\;x=1,\qquad A_2:\;y=1,\qquad A_3:\;x+y=n+1,
$$  
and let  
$$
B = B_1\cup B_2\cup B_3,\quad 
\begin{cases}
B_1 = \{(1,b):1\le b\le n\},\\
B_2 = \{(a,1):1\le a\le n\},\\
B_3 = \{(a,n+1-a):1\le a\le n\}.
\end{cases}
$$  
$B$ is the set of points on the boundary of the triangular region containing $S_n$; it contains $3n-3$ distinct points.

**Lemma.** For $n\ge 4$, any covering of $S_n$ by $n$ lines contains at least one of the special lines $A_1,A_2,A_3$.

*Proof.* Suppose none of $A_1,A_2,A_3$ is used. We show that every line in the covering contains at most two points of $B$.

- A vertical line $x=c$ with $c\ne1$: it meets $B_2$ at most at $(c,1)$ (if $1\le c\le n$) and meets $B_3$ at most at $(c,n+1-c)$; at most two points.
- A horizontal line $y=c$ with $c\ne1$: analogous, at most two points.
- A line parallel to $x+y=n+1$, i.e. $x+y=t$ with $t\ne n+1$: it meets $B_1$ at most at $(1,t-1)$ (if $2\le t\le n+1$) and $B_2$ at most at $(t-1,1)$; at most two points.
- A sunny line (slope not $0,\infty,-1$): it can intersect each of the three sides $x=1$, $y=1$, $x+y=n+1$ in at most one point; moreover a straight line cannot intersect all three sides of a triangle, so it contains at most two points of $B$.

Hence each of the $n$ lines contributes at most two incidences with $B$. Consequently the total number of incidences is at most $2n$. But every point of $B$ (there are $3n-3$ of them) must be incident to at least one line, giving $2n\ge 3n-3$, i.e. $n\le3$, a contradiction. Therefore at least one special line is present. ∎

---

### 3.  Base case $n=3$

We prove directly that for $S_3$ any covering by three lines uses exactly 0, 1, or 3 sunny lines.

Let the three lines be $\ell_1,\ell_2,\ell_3$. Write $s$ for the number of sunny lines among them. Since there are only three lines, $s$ can be 0,1,2,3. We must exclude $s=2$.

Assume, for contradiction, that there exists a covering with exactly two sunny lines. Then one line, call it $L$, is non‑sunny (vertical, horizontal, or slope $-1$).  
First we examine how many points of $S_3$ a sunny line can contain. A straightforward check of the six points  
$$
A=(1,1),\; B=(1,2),\; C=(1,3),\; D=(2,1),\; E=(2,2),\; F=(3,1)
$$  
shows that the only lines containing three points are the non‑sunny lines $x=1$, $y=1$ and $x+y=4$; hence **every sunny line contains at most two points**. Consequently a line that is not sunny covers at most three points (and actually the three point‑lines are exactly the non‑sunny ones).  

If $L$ contains at most one point of $S_3$, then the two sunny lines together cover at most $2+2=4$ points, so $4+1=5$ points in total are covered, but $S_3$ has six points – impossible. Therefore $L$ must contain at least two points. The possible (distinct) non‑sunny lines containing at least two points from $S_3$ are exactly the following six:

1. $x=1$ (covers $A,B,C$) – three points  
2. $y=1$ (covers $A,D,F$) – three points  
3. $x+y=4$ (covers $C,E,F$) – three points  
4. $x=2$ (covers $D,E$) – two points  
5. $y=2$ (covers $B,E$) – two points  
6. $x+y=3$ (covers $B,D$) – two points  

We show that in each case the remaining points cannot be covered by the two sunny lines.

* **Cases 1–3 (the “sides” with three points).**  

  * **Case 1:** $L=x=1$ covers $\{A,B,C\}$.  
    The uncovered points are $D,E,F$. Notice that any two of $\{D,E,F\}$ lie on a non‑sunny line:  
    $(D,E)$ on $x=2$, $(D,F)$ on $y=1$, $(E,F)$ on $x+y=4$.  
    Hence a sunny line can contain **at most one** of $\{D,E,F\}$ (if it contained two, it would coincide with that non‑sunny line). With only two sunny lines, at most two of the three points can be covered – contradiction.

  * **Case 2:** $L=y=1$ covers $\{A,D,F\}$.  
    Uncovered: $B,C,E$. Any two of $\{B,C,E\}$ lie on a non‑sunny line: $(B,C)$ on $x=1$, $(B,E)$ on $y=2$, $(C,E)$ on $x+y=4$. Thus a sunny line can contain at most one of $\{B,C,E\}$; two sunny lines cannot cover three points – contradiction.

  * **Case 3:** $L=x+y=4$ covers $\{C,E,F\}$.  
    Uncovered: $A,B,D$. Any two of $\{A,B,D\}$ lie on a non‑sunny line: $(A,B)$ on $x=1$, $(A,D)$ on $y=1$, $(B,D)$ on $x+y=3$. Again, a sunny line contains at most one of them; contradiction.

* **Cases 4–6 (the lines with two points).**  

  * **Case 4:** $L=x=2$ covers $\{D,E\}$.  
    Uncovered: $A,B,C,F$. Points $A,B,C$ all lie on the vertical line $x=1$ (non‑sunny). Consequently a sunny line can contain at most one of $\{A,B,C\}$. Two sunny lines can cover at most two of these three points, leaving at least one uncovered – contradiction.

  * **Case 5:** $L=y=2$ covers $\{B,E\}$.  
    Uncovered: $A,C,D,F$. Points $A,D,F$ lie on $y=1$ (non‑sunny). A sunny line may contain at most one of $\{A,D,F\}$, so two sunny lines cover at most two of them – contradiction.

  * **Case 6:** $L=x+y=3$ covers $\{B,D\}$.  
    Uncovered: $A,C,E,F$. Points $C,E,F$ lie on $x+y=4$ (non‑sunny). A sunny line contains at most one of $\{C,E,F\}$, thus two sunny lines cover at most two of them – contradiction.

All possibilities lead to a contradiction. Hence a covering with exactly two sunny lines cannot exist, so for $n=3$ we have $s\in\{0,1,3\}$. (Existence of coverings for these values was shown in Section 1.)

---

### 4.  Inductive step ($n\ge 4$)

Assume that for every integer $m<n$ any covering of $S_m$ by $m$ lines uses $0$, $1$ or $3$ sunny lines.  
Consider a covering $\mathcal{L}$ of $S_n$ by $n$ lines, and let $s$ be the number of sunny lines in $\mathcal{L}$. By Lemma 2.1, $\mathcal{L}$ contains at least one of the special lines $A_1$, $A_2$, $A_3$. We treat the three symmetric cases.

**Case A:** $A_1:\,x=1$ belongs to $\mathcal{L}$.  
Remove this line and all points with $x=1$ from the consideration. The remaining points are  
$$
S' = \{(x,y)\in S_n \mid x\ge 2\}.
$$  
Define the translation $\varphi:(x,y)\mapsto (x-1,\,y)$. Then $\varphi$ is a bijection from $S'$ onto $S_{n-1}$ because  
$$
2\le x,\; y\ge1,\; x+y\le n+1 \;\Longleftrightarrow\; 1\le x-1,\; y\ge1,\; (x-1)+y\le n.
$$  
For every line $\ell\in\mathcal{L}\setminus\{A_1\}$ the image $\varphi(\ell)$ is again a line (translation parallel to the $x$-axis preserves linearity) and the slope is unchanged, so $\varphi(\ell)$ is sunny iff $\ell$ is sunny. Moreover $\{\varphi(\ell)\mid\ell\in\mathcal{L}\setminus\{A_1\}\}$ covers $S_{n-1}$ because for any $p\in S_{n-1}$ there is a pre‑image $q\in S'$ with $\varphi(q)=p$ and, as $q$ is covered by some $\ell\neq A_1$, the point $p$ is covered by $\varphi(\ell)$. Thus we obtain a covering of $S_{n-1}$ by $n-1$ lines with exactly $s$ sunny lines. By the induction hypothesis $s\in\{0,1,3\}$.

**Case B:** $A_2:\,y=1$ belongs to $\mathcal{L}$.  
The argument is completely analogous (translate in the $y$-direction).

**Case C:** $A_3:\,x+y=n+1$ belongs to $\mathcal{L}$.  
Remove this line and all points on it. The remaining points satisfy $x,y\ge 1$ and $x+y\le n$, i.e. they form exactly $S_{n-1}$. Notice that no point of $S_{n-1}$ lies on $A_3$ because $a+b=n+1$ would contradict $a+b\le n$. Hence the other $n-1$ lines of $\mathcal{L}$ still cover $S_{n-1}$, and the number of sunny lines among them is again $s$. By the induction hypothesis $s\in\{0,1,3\}$.

In every case we conclude $s\in\{0,1,3\}$, completing the induction.

---

### 5.  Conclusion

For every integer $n\ge 3$ the only non‑negative integers $k$ for which there exists a covering of $S_n$ by $n$ distinct lines with exactly $k$ sunny lines are  

$$
\boxed{0,\;1,\;3}.
$$