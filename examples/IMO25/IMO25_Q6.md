### 1. Upper bound: $2n-2$ tiles suffice

Place the holes on the main diagonal, i.e., $\pi(i)=i$ for all $i$.  
For each $i=1,\dots,n-1$ place a horizontal rectangle covering row $i$, columns $i+1$ to $n$.  
For each $j=1,\dots,n-1$ place a vertical rectangle covering column $j$, rows $j+1$ to $n$.  

These $2n-2$ rectangles are disjoint: a horizontal rectangle lies in a row $i$ and columns $>i$; a vertical rectangle lies in a column $j$ and rows $>j$. Their intersection would require a cell $(i,j)$ with $i>j$ (for the vertical) and simultaneously $i<j$ (for the horizontal), which is impossible. Every cell $(i,j)$ with $i\neq j$ is covered by exactly one of these rectangles. Hence $2n-2$ tiles form an admissible covering.

---

### 2. Lower bound: every admissible covering uses at least $2n-2$ tiles

We prove by induction on $n$.

**Base cases:** $n=1$ (no covered squares, $0=2\cdot1-2$) and $n=2$ (the two uncovered squares are opposite corners; they must be covered by two separate rectangles, and $2=2\cdot2-2$). The statement holds.

**Induction step.** Assume $n\ge 3$ and the statement is true for all boards of size $< n$. Consider an admissible covering of the $n\times n$ board with $m$ rectangles. We may assume the covering is **minimal**, i.e., no covering of the same board with the same hole set uses fewer rectangles. (If the original covering is not minimal, we replace it by a smaller one; a bound for minimal coverings implies the bound for all.)

#### Lemma 1 (Relabeling)
Without loss of generality we may assume the hole in the first row is at column 1, i.e., $\pi(1)=1$.

*Proof.* Apply a permutation of the columns that sends the column containing the hole of row 1 to column 1. This relabeling preserves the grid structure and the number of rectangles, and yields a new permutation with $\pi(1)=1$. ∎

From now on we assume $\pi(1)=1$; thus the cell $(1,1)$ is a hole.

#### Lemma 2 (Existence of a row‑1 rectangle)
In the minimal covering under consideration, there exists at least one rectangle that lies completely in the first row.

*Proof.* Suppose, for contradiction, that every rectangle intersecting row 1 has height at least 2 (i.e., extends to row 2 or below).  
Let $S$ be the set of rectangles that intersect row 1. For each $R\in S$, denote by $h(R)$ the greatest row index covered by $R$; because the rectangle meets row 1, its rows form a contiguous interval $[1, h(R)]$. By our supposition, $h(R)\ge 2$ for all $R\in S$.

Choose $R_0\in S$ with the smallest value of $h(R)$; set $m = h(R_0)\ge 2$. Let the column interval of $R_0$ be $[\ell, r]$. Since $(1,1)$ is a hole, we have $\ell\ge 2$.  

Now consider the hole at $(m, \pi(m))$. This hole is not covered by $R_0$; otherwise $(1,\pi(m))$ (which is in row 1) would also be covered, contradicting that the only hole in row 1 is at column 1. Therefore $\pi(m)\notin[\ell, r]$.

Row 1 column $\pi(m)$ is covered (the only hole in row 1 is at column 1), so there exists a rectangle $R_1\in S$ covering $(1,\pi(m))$. Because $\pi(m)\notin[\ell, r]$, the rectangles $R_0$ and $R_1$ are distinct.

If $h(R_1)\ge m$, then $R_1$ would cover rows $1,\dots,h(R_1)$, hence row $m$, and its column interval contains $\pi(m)$, so it would also cover $(m,\pi(m))$ – a contradiction. Thus $h(R_1) < m$.

Now $R_1\in S$ and $h(R_1) < m$. If $h(R_1)=1$, then $R_1$ is completely contained in row 1, contradicting our supposition directly. If $h(R_1)\ge 2$, then we have found a rectangle in $S$ with a smaller bottom row than $m$, contradicting the minimality of $m$.  

In either case we reach a contradiction, so our initial assumption must be false. Hence there exists at least one rectangle in row 1. ∎

#### Symmetric lemma (column‑1 rectangle)
Swapping the roles of rows and columns (or repeating the same argument on the transposed board) shows that there also exists a rectangle completely contained in the first column.

**Note:** The row‑1 rectangle and the column‑1 rectangle are distinct. Indeed, the row‑1 rectangle cannot contain column 1 (otherwise it would cover the hole $(1,1)$), and the column‑1 rectangle cannot contain row 1. Therefore they are disjoint.

#### Deleting the first row and first column
Remove the entire first row and first column from the board. The two rectangles identified above disappear completely (they were confined to row 1 and column 1 respectively). All other rectangles are either:
- completely contained in rows $2,\dots,n$ and columns $2,\dots,n$ (unchanged), or
- intersected the first row or first column but not both; after deletion they are cropped to rectangles on the $(n-1)\times (n-1)$ subboard (still axis‑aligned and disjoint).

The resulting set of rectangles covers exactly those squares of the subboard that were covered originally. The holes in the subboard are the pairs $(i-1,\pi(i)-1)$ for $i=2,\dots,n$, which form a permutation of size $n-1$. Hence we obtain an admissible covering of an $(n-1)\times (n-1)$ board.

Let $m$ be the number of rectangles in the original covering. Since at least the two border rectangles are removed, the new covering uses at most $m-2$ rectangles. By the induction hypothesis, any admissible covering of an $(n-1)\times (n-1)$ board requires at least $2(n-1)-2 = 2n-4$ rectangles. Therefore

$$
m-2 \; \ge \; 2n-4 \qquad\Longrightarrow\qquad m \; \ge \; 2n-2 .
$$

This completes the inductive proof of the lower bound.

---

### 3. Conclusion for $n=2025$

The construction in 2.1 gives an example with $2\cdot2025-2 = 4048$ tiles. The lower bound proves that no admissible covering can use fewer than $4048$ tiles. Hence the minimum possible number of tiles is exactly

$$
\boxed{4048}.
$$