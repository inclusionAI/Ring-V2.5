**Notation.** For $n\in\mathbb{N}$ let  

$$
S_n = \sum_{i=1}^{n}x_i,\qquad Q_n = \sum_{i=1}^{n}x_i^2.
$$

On an odd turn $n$ (Alice) she must choose $x_n\ge0$ with $S_n\le \lambda n$; on an even turn $n$ (Bazza) he must choose $x_n\ge0$ with $Q_n\le n$. If a player cannot move, the opponent wins.  

For $k\ge0$ define the **slack** after Bazza’s turn $2k$ by  

$$
A_k = \lambda(2k+1)-S_{2k}.
$$

This is the maximal $x$ Alice may legally play at turn $2k+1$.  

**Lemma.** For any $c$ with $0\le c\le\sqrt{2}$,

$$
c+\sqrt{2-c^2} \ge \sqrt{2},
$$

with equality exactly for $c=0$ or $c=\sqrt{2}$.  

*Proof.* Squaring gives $2+2c\sqrt{2-c^2}\ge2$, so the left side is $\ge\sqrt{2}$. Equality iff $c\sqrt{2-c^2}=0$, i.e. $c=0$ or $c=\sqrt{2}$. ∎  

---

#### Case 1: $\lambda > \dfrac{\sqrt{2}}{2}$ – Alice wins  

**Alice’s strategy.**  
Before her turn $2k+1$ ($k\ge0$) compute  

$$
T_k = \sqrt{2k+2 - Q_{2k}}.
$$

* If $A_k > T_k$, choose any $x$ with $T_k < x \le A_k$ (e.g. $x = \frac{A_k+T_k}{2}$).  
  Then  

  $$
  S_{2k}+x \le S_{2k}+A_k = \lambda(2k+1),
  $$
  so the sum constraint holds; moreover  

  $$
  Q_{2k+1} = Q_{2k}+x^2 > Q_{2k}+T_k^2 = 2k+2,
  $$
  which makes Bazza’s next move impossible – Alice wins.  

* Otherwise ($A_k \le T_k$) she plays $x=0$. Since $A_k\ge0$ (see below), this is legal.

We prove that this strategy guarantees a win.  

Because Alice plays $0$ until she wins, for every $k$ reached before winning we have $S_{2k}$ equal to the sum of Bazza’s moves and $Q_{2k}$ equal to the sum of their squares. The rules give $Q_{2k}\le 2k$.  

*Lower bound for $A_k$:* By the Cauchy–Schwarz inequality  

$$
S_{2k}^2 \le k\cdot Q_{2k} \le k\cdot 2k = 2k^2\quad\Longrightarrow\quad S_{2k}\le k\sqrt{2}.
$$

Hence  

$$
A_k = \lambda(2k+1)-S_{2k} \ge \lambda(2k+1)-k\sqrt{2} = k(2\lambda-\sqrt{2})+\lambda. \tag{1}
$$

*Upper bound for $T_k$:* Clearly  

$$
T_k = \sqrt{2k+2-Q_{2k}} \le \sqrt{2k+2}. \tag{2}
$$

Since $\lambda > \sqrt{2}/2$, we have $2\lambda-\sqrt{2}>0$. The right‑hand side of (1) grows linearly in $k$ while $\sqrt{2k+2}$ grows like $\sqrt{k}$. Thus there exists an integer $K$ such that for all $k\ge K$  

$$
k(2\lambda-\sqrt{2})+\lambda \;>\; \sqrt{2k+2}. \tag{3}
$$

For such a $k$, using (1) and (2),  

$$
A_k \ge k(2\lambda-\sqrt{2})+\lambda \;>\; \sqrt{2k+2} \ge T_k,
$$

so the winning condition $A_k>T_k$ holds. Consequently, by turn $2K+1$ at the latest Alice will win.  

It remains to see that before that turn she never loses.  
From (1) we have $A_k \ge \lambda>0$ for every $k$; thus playing $0$ is always legal. Also, because $A_k>0$, the sum constraint is satisfied after playing $0$. The game therefore continues until she meets the winning condition.  

Hence Alice has a winning strategy for all $\lambda > \sqrt{2}/2$.  

---

#### Case 2: $0 < \lambda < \dfrac{\sqrt{2}}{2}$ – Bazza wins  

**Bazza’s strategy.**  
On each even turn $2k$ ($k\ge1$) he plays the maximal allowed number  

$$
x_{2k} = \sqrt{\,2k - Q_{2k-1}\,}. \tag{4}
$$

We show that with this strategy Alice eventually cannot move and that she never wins earlier.  

Let $a_k = x_{2k-1}$ (Alice’s choice on turn $2k-1$). Throughout the proof we work under the assumption that the game has not ended before that turn; in particular the move is legal.  

We prove by induction on $k$ that:  

*After Bazza’s move on turn $2k$ we have*  

$$
Q_{2k}=2k, \tag{5}
$$  

*and the total sum satisfies*  

$$
S_{2k} \ge k\sqrt{2}. \tag{6}
$$

*Base $k=1$:* Before turn 1, $S_0=Q_0=0$. Alice chooses some $a_1\ge0$ with $a_1\le \lambda$ (because $A_0=\lambda$). Since $\lambda<\sqrt{2}/2<\sqrt{2}$, we have $a_1^2<2$. Bazza then takes  

$$
x_2 = \sqrt{2 - a_1^2},
$$

which is possible because $Q_1 = a_1^2 \le 2$. Then $Q_2 = a_1^2 + (2-a_1^2)=2$.  
The increase in the sum during the first round is  

$$
\Delta S_1 = a_1 + \sqrt{2-a_1^2} \ge \sqrt{2}
$$

by the Lemma, with equality only if $a_1=0$ or $a_1=\sqrt{2}$ (the latter is impossible). Hence $S_2 \ge \sqrt{2}$, establishing (5) and (6) for $k=1$.  

*Inductive step:* Assume (5) and (6) hold for a given $k$.  
Before turn $2k+1$ we have $S_{2k}$ and $Q_{2k}=2k$. Alice’s slack is  

$$
A_k = \lambda(2k+1)-S_{2k}.
$$

From (6), $S_{2k}\ge k\sqrt{2}$, so  

$$
A_k \le \lambda(2k+1)-k\sqrt{2} = \lambda + k(2\lambda-\sqrt{2}). \tag{7}
$$

Because $\lambda < \sqrt{2}/2$, we have $2\lambda-\sqrt{2}<0$; the right‑hand side of (7) is at most $\lambda$ (it decreases when $k$ increases). Hence  

$$
A_k \le \lambda < \frac{\sqrt{2}}{2} < \sqrt{2}. \tag{8}
$$

Alice chooses $a_{k+1}\ge0$ with $a_{k+1} \le A_k$. Then $a_{k+1}^2 \le A_k^2 < 2$.  
Now Bazza’s move at turn $2k+2$: using (4),  

$$
x_{2k+2} = \sqrt{\,2(k+1) - Q_{2k+1}\,},\qquad Q_{2k+1}=Q_{2k}+a_{k+1}^2 = 2k + a_{k+1}^2.
$$

Since $a_{k+1}^2 < 2$, the radicand $2(k+1)-(2k+a_{k+1}^2)=2 - a_{k+1}^2$ is positive, and  

$$
x_{2k+2} = \sqrt{2 - a_{k+1}^2}.
$$

Consequently,  

$$
Q_{2k+2}= Q_{2k+1} + x_{2k+2}^2 = (2k + a_{k+1}^2) + (2 - a_{k+1}^2) = 2(k+1),
$$

which proves (5) for $k+1$.  

The increase in the total sum during round $k+1$ is  

$$
\Delta S_{k+1} = a_{k+1} + \sqrt{2 - a_{k+1}^2} \ge \sqrt{2},
$$

again by the Lemma. Adding to $S_{2k}$ gives  

$$
S_{2k+2} = S_{2k} + \Delta S_{k+1} \ge k\sqrt{2} + \sqrt{2} = (k+1)\sqrt{2},
$$

establishing (6) for $k+1$. This completes the induction.  

From (6) we have $S_{2k} \ge k\sqrt{2}$ for all $k$. Alice can only move at turn $2k+1$ if  

$$
S_{2k} \le \lambda(2k+1). \tag{9}
$$

Because $\lambda < \sqrt{2}/2$, the inequality  

$$
k\sqrt{2} \le \lambda(2k+1)
$$

is equivalent to  

$$
k(\sqrt{2}-2\lambda) \le \lambda.
$$

Since $\sqrt{2}-2\lambda >0$, the left‑hand side grows without bound, so there exists a finite $k$ for which it fails. For that $k$, (9) is false; i.e., $S_{2k} > \lambda(2k+1)$. Then at turn $2k+1$ Alice cannot choose any $x\ge0$ satisfying $S_{2k}+x\le \lambda(2k+1)$ – she loses.  

Finally, we verify that Alice never wins before that moment. To win on turn $2k+1$ she would have to play a number $a_{k+1}$ such that after her move Bazza cannot move on turn $2k+2$. This requires  

$$
Q_{2k+1} = Q_{2k} + a_{k+1}^2 > 2k+2.
$$

Since $Q_{2k}=2k$ (by (5)), this is equivalent to $a_{k+1}^2 > 2$. But (8) shows $A_k < \sqrt{2}$, and because $a_{k+1}\le A_k$, we have $a_{k+1}<\sqrt{2}$ and thus $a_{k+1}^2<2$. Hence Alice can never force an immediate win.  

Therefore Bazza’s strategy guarantees his victory for all $\lambda < \sqrt{2}/2$.  

---

#### Case 3: $\lambda = \dfrac{\sqrt{2}}{2}$ – Neither has a winning strategy  

We exhibit drawing strategies for both players.  

*Alice can force a draw.*  
She simply always plays $0$.  
- Before each of her turns (odd index $2k+1$) the sum $S_{2k}$ consists only of Bazza’s numbers. By Cauchy–Schwarz,  

$$
S_{2k}^2 \le k\cdot Q_{2k} \le k\cdot 2k = 2k^2 \quad\Longrightarrow\quad S_{2k} \le k\sqrt{2}.
$$  

Since $\lambda = \sqrt{2}/2$,  

$$
\lambda(2k+1) = \frac{\sqrt{2}}{2}(2k+1) = k\sqrt{2} + \frac{\sqrt{2}}{2} > k\sqrt{2} \ge S_{2k}.
$$  

Thus her sum constraint is satisfied (she plays $0$).  
- After her move $Q$ is unchanged and remains $\le 2k$. At the following even turn Bazza’s constraint is $Q_{2k+1}+x_{2k+2}^2\le 2k+2$. Because $Q_{2k+1}\le 2k$, we have $2k+2-Q_{2k+1}\ge2$, so Bazza can always choose a legal number (e.g. the maximum $\sqrt{2k+2-Q_{2k+1}}$).  
Hence the game never terminates; Alice never loses.  

*Bazza can force a draw.*  
He adopts the maximal strategy from Case 2: on turn $2k$ play $x_{2k} = \sqrt{2k-Q_{2k-1}}$.  
- From the proof of Case 2 we obtain that the slack before Alice’s turn $2k+1$ satisfies (cf. (7))  

$$
A_k \le \lambda + k(2\lambda-\sqrt{2}) = \frac{\sqrt{2}}{2} + k\cdot 0 = \frac{\sqrt{2}}{2}.
$$  

Thus $A_k \le \sqrt{2}/2 < \sqrt{2}$ for all $k$.  
- To win on her turn Alice would need to play a number $x$ with $x^2>2$ (so that $Q$ exceeds the next bound). This requires $x>\sqrt{2}$ and therefore $A_k > \sqrt{2}$, which is impossible.  
- Moreover, by construction Bazza always respects his own quadratic constraint.  

Consequently, when Alice uses her “always $0$” strategy and Bazza uses his maximal strategy, the game continues forever; in particular, neither loses.  

Because each player has a strategy to avoid losing (a drawing strategy), neither possesses a winning strategy when $\lambda = \sqrt{2}/2$.  

---

**Conclusion**  

$$
\boxed{
\begin{array}{ll}
\text{Alice has a winning strategy for} & \lambda > \dfrac{\sqrt{2}}{2}, \\
\text{Bazza has a winning strategy for} & 0 < \lambda < \dfrac{\sqrt{2}}{2}, \\
\text{For } \lambda = \dfrac{\sqrt{2}}{2} \text{ neither player has a winning strategy.}
\end{array}
}
$$