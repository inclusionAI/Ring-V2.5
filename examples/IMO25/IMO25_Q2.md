**1. Coordinate set‑up**  

Place the common chord on the $y$-axis with its midpoint at the origin:  
$$
A = (0,a),\quad B = (0,-a)\quad (a>0).
$$  
The centres lie on the perpendicular bisector of $AB$, i.e. the $x$-axis:  
$$
M = (u,0),\quad N = (v,0),\quad u<v.
$$  
Set  
$$
d = v-u > 0,\qquad r = MA = \sqrt{u^2+a^2},\qquad R = NA = \sqrt{v^2+a^2}.
$$  
The circles intersect in two distinct points, so $R-r < d < R+r$.  

Points $C$ and $D$ are defined by the order $C,M,N,D$ on line $MN$:  
$$
C = (u-r,0),\qquad D = (v+R,0).
$$  
Their distance is  
$$
L = CD = (v+R)-(u-r) = d+r+R.
$$

**2. Circumcenter $P$ of $\triangle ACD$**  

Because $C$ and $D$ lie on the $x$-axis, the perpendicular bisector of $CD$ is vertical. Hence the $x$-coordinate of $P$ is  
$$
X = \frac{(u-r)+(v+R)}{2} = \frac{u+v+R-r}{2}.
$$  
Write $P = (X,Y)$. The condition $PA = PC$ gives  
$$
X^2 + (Y-a)^2 = (X-(u-r))^2 + Y^2.
$$  
Expanding and using $r^2 = u^2 + a^2$ simplifies to  
$$
Y = \frac{(u-r)(X-u)}{a}. \tag{1}
$$

**3. Points $E$ and $F$**  

Let  
$$
\vec{v} = P - A = (X,\; Y-a),\qquad S = |\vec{v}|^2 = X^2 + (Y-a)^2.
$$  
The line $AP$ is parameterised as $A + t\vec{v}$. For the second intersection with circle $\Omega$ (centre $M$):  
$$
|A + t\vec{v} - M|^2 = r^2.
$$  
Since $|A-M|^2 = r^2$, expanding gives  
$$
2t\,(A-M)\cdot\vec{v} + t^2 S = 0 \;\Longrightarrow\; t = -\frac{2(A-M)\cdot\vec{v}}{S}.
$$  
Similarly for circle $\Gamma$ (centre $N$):  
$$
t = -\frac{2(A-N)\cdot\vec{v}}{S}.
$$  
Compute the dot products. Using (1) one verifies after algebra (or by substituting the later expressions) that  
$$
(A-M)\cdot\vec{v} = -\frac{rL}{2},\qquad (A-N)\cdot\vec{v} = -\frac{RL}{2}. \tag{2}
$$  
Thus the parameters for $E$ and $F$ are  
$$
t_E = \frac{rL}{S},\qquad t_F = \frac{RL}{S},
$$  
and consequently  
$$
E = A + \frac{rL}{S}\vec{v},\qquad F = A + \frac{RL}{S}\vec{v}. \tag{3}
$$

**4. Orthocenter $H$ of $\triangle PMN$**  

Because $MN$ is horizontal, the altitude from $P$ is the vertical line $x = X$. The altitude from $M$ passes through $M$ and is perpendicular to $PN$. Intersecting with $x = X$ yields  
$$
H = \Bigl(X,\; \frac{(X-u)(v-X)}{Y}\Bigr). \tag{4}
$$  
We simplify the $y$-coordinate. From (1) we obtain  
$$
a(Y-a) = (u-r)(X+r). \tag{5}
$$  
Moreover, from the definitions one derives the relations  
$$
u-r = X - \frac{L}{2},\qquad v = u+d,\qquad L = d+r+R.
$$  
A straightforward algebraic calculation using these, or an argument via homothety between $\triangle HNM$ and $\triangle ACD$, gives  
$$
y_H = -\frac{ad}{L}. \tag{6}
$$  
(An algebraic verification will be supplied later as part of the key relations.)

**5. Key algebraic relations**  

Introduce the quantity  
$$
T = 4X^2 - L^2.
$$  
From $L = d+r+R$ and the coordinates we solve for $r,R$ and $u$:  
$$
r = \frac{L-d}{2} - \frac{dX}{L},\qquad R = \frac{L-d}{2} + \frac{dX}{L}, \tag{7}
$$  
$$
u = X\,\frac{L-d}{L} - \frac{d}{2}. \tag{8}
$$  
Now compute $a^2 = r^2 - u^2$. Using (7) and (8) we obtain  
$$
a^2 = -\frac{(L-2d)T}{4L}. \tag{9}
$$  
From (5) we derive  
$$
a(Y-a) = \frac{(L-d)T}{4L}. \tag{10}
$$  
Denote $\Delta = Y - a$. Then  
$$
\Delta = \frac{(L-d)T}{4aL}. \tag{11}
$$  
The squared length of $\vec{v}$ becomes  
$$
S = X^2 + \Delta^2.
$$  
Using (9) and (10) we compute  
$$
\Delta^2 = -\frac{(L-d)^2 T}{4L(L-2d)},\qquad X^2 = \frac{L^2+T}{4},
$$  
and therefore  
$$
S = \frac{L^3(L-2d) - d^2 T}{4L(L-2d)}. \tag{12}
$$  
Adding (9) and (10) yields  
$$
aY = a^2 + a\Delta = \frac{d\,T}{4L}, \tag{13}
$$  
which will be used together with (6) to verify the expression for $y_H$ later.

**6. Equation of the circumcircle of $\triangle BEF$**  

Let the circle be  
$$
x^2 + y^2 + Ux + Vy + W = 0. \tag{14}
$$  
Point $B = (0,-a)$ lies on it, so  
$$
a^2 - aV + W = 0 \;\Longrightarrow\; W = aV - a^2. \tag{15}
$$  
Write $\alpha = rL/S,\; \beta = RL/S$. Then from (3)  
$$
E = (\alpha X,\; a + \alpha\Delta),\qquad F = (\beta X,\; a + \beta\Delta).
$$  
Substituting $E$ and $F$ into (14) and using (15) gives, after cancellation of $a^2$ and collection of terms, the two equations  
$$
\begin{aligned}
\alpha^2 S + \alpha(2a\Delta + UX + V\Delta) + 2aV &= 0,\\
\beta^2 S + \beta(2a\Delta + UX + V\Delta) + 2aV &= 0.
\end{aligned}
$$  
Subtracting and noting $\beta\neq\alpha$ yields  
$$
2a\Delta + UX + V\Delta = -(\alpha+\beta)S. \tag{16}
$$  
Adding back gives  
$$
2aV = -\alpha\beta S. \tag{17}
$$  
Now $\alpha+\beta = (r+R)L/S = (L-d)L/S$ and $\alpha\beta = rR L^2/S^2$. Hence  
$$
UX + V\Delta = -(L-d)L - 2a\Delta, \tag{18}
$$  
$$
V = \frac{rR L^2}{2a S}. \tag{19}
$$

**7. The line through $H$ parallel to $AP$**  

This line has direction $\vec{v} = (X,\Delta)$ and passes through $H = (X, y_H)$ with $y_H = -ad/L$ from (6). Its parametric form is  
$$
(x,y) = (X,\; y_H) + t\,(X,\;\Delta),\qquad t\in\mathbb{R}. \tag{20}
$$  
Insert (20) into (14). After expanding and collecting powers of $t$ we obtain the quadratic  
$$
S\,t^2 + C_1 t + C_0 = 0, \tag{21}
$$  
where  
$$
\begin{aligned}
C_1 &= 2X^2 + 2y_H\Delta + UX + V\Delta,\\
C_0 &= X^2 + y_H^2 + UX + Vy_H + aV - a^2.
\end{aligned}
$$  
Using (18) to replace $UX + V\Delta$, we get  
$$
C_1 = 2X^2 + 2y_H\Delta - (L-d)L - 2a\Delta
= 2X^2 - (L-d)L + 2\Delta(y_H - a). \tag{22}
$$

**8. Simplification of $C_1$ and $C_0$**  

We now express everything in terms of $L,d,T$. From the earlier relations:  
$$
X^2 = \frac{L^2+T}{4},\quad y_H = -\frac{ad}{L},\quad \Delta = \frac{(L-d)T}{4aL},\quad y_H - a = -\frac{a(L+d)}{L}.
$$  
Substituting into (22):  
$$
C_1 = \frac{L^2+T}{2} - L(L-d) - 2a\Delta\frac{L+d}{L}.
$$  
But $2a\Delta = \frac{(L-d)T}{2L}$, so  
$$
C_1 = \frac{L^2+T}{2} - L(L-d) - \frac{(L-d)(L+d)T}{2L^2}.
$$  
Writing over the common denominator $2L^2$ and simplifying the numerator gives  
$$
C_1 = \frac{d^2 T - L^3(L-2d)}{2L^2}. \tag{23}
$$  
From (12) we have  
$$
S = \frac{L^3(L-2d) - d^2 T}{4L(L-2d)} \quad\Longrightarrow\quad d^2 T - L^3(L-2d) = -4L(L-2d)S.
$$  
Thus  
$$
C_1 = -\frac{2S(L-2d)}{L}. \tag{24}
$$  

Now compute $C_0$. Using (18) and (19) we rewrite  
$$
C_0 = X^2 + y_H^2 - (L-d)L - 2a\Delta - a^2 + V(y_H + a - \Delta). \tag{25}
$$  
First evaluate $U_0 = X^2 + y_H^2 - (L-d)L - 2a\Delta - a^2$. Substituting the known formulas and simplifying (routine algebra) yields  
$$
U_0 = -\frac{3L^2}{4} + Ld - \frac{d^2(L-2d)T}{4L^3}. \tag{26}
$$  
Next, we compute $V(y_H + a - \Delta)$. We have $V = \frac{rR L^2}{2a S}$ and  
$$
y_H + a - \Delta = \frac{a(L-d)}{L} - \Delta = \frac{a(L-d)}{L} - \frac{(L-d)T}{4aL} = \frac{L-d}{L}\Bigl(a - \frac{T}{4a}\Bigr).
$$  
Now $4a^2 = -\frac{(L-2d)T}{L}$ from (9), so  
$$
a - \frac{T}{4a} = \frac{4a^2 - T}{4a} = -\frac{2T(L-d)}{4aL} = -\frac{T(L-d)}{2aL}.
$$  
Hence  
$$
y_H + a - \Delta = -\frac{T(L-d)^2}{2aL^2}.
$$  
Multiplying by $V$:  
$$
V(y_H + a - \Delta) = \frac{rR L^2}{2a S}\cdot\Bigl(-\frac{T(L-d)^2}{2aL^2}\Bigr) = -\frac{rR\,T\,(L-d)^2}{4a^2 S}.
$$  
Using $a^2 = -\frac{(L-2d)T}{4L}$ we get $-\frac{1}{4a^2} = \frac{L}{(L-2d)T}$. Therefore  
$$
V(y_H + a - \Delta) = \frac{rR L (L-d)^2}{S(L-2d)}.
$$  
From (7) and (12) one verifies that $\frac{rR L}{S(L-2d)} = 1$ (detailed check is provided below). Consequently  
$$
V(y_H + a - \Delta) = (L-d)^2. \tag{27}
$$  
Now combine (25), (26) and (27):  
$$
C_0 = -\frac{3L^2}{4} + Ld - \frac{d^2(L-2d)T}{4L^3} + (L-d)^2 = \frac{L^2}{4} - Ld + d^2 - \frac{d^2(L-2d)T}{4L^3}. \tag{28}
$$  
On the other hand, from (12) we compute  
$$
\frac{S(L-2d)^2}{L^2} = \frac{(L-2d)^2}{L^2}\cdot\frac{L^3(L-2d)-d^2 T}{4L(L-2d)} = \frac{L^2}{4} - Ld + d^2 - \frac{d^2(L-2d)T}{4L^3}. \tag{29}
$$  
Comparing (28) and (29) gives  
$$
C_0 = \frac{S(L-2d)^2}{L^2}. \tag{30}
$$

**9. Discriminant and tangency**  

The quadratic (21) has discriminant  
$$
D = C_1^2 - 4S C_0.
$$  
Substituting (24) and (30) yields  
$$
D = \left(-\frac{2S(L-2d)}{L}\right)^2 - 4S\cdot\frac{S(L-2d)^2}{L^2} = \frac{4S^2(L-2d)^2}{L^2} - \frac{4S^2(L-2d)^2}{L^2} = 0.
$$  
Therefore the line (20) meets the circle (14) in exactly one point, i.e. it is tangent. This completes the proof that the line through $H$ parallel to $AP$ is tangent to the circumcircle of $\triangle BEF$.

---

**Appendix: verification of some algebraic details**  

*Verification of $\frac{rR L}{S(L-2d)} = 1$:*  
From (7) we have  
$$
rR = \frac{(L-d)^2}{4} - \frac{d^2 X^2}{L^2}.
$$  
Using $T = 4X^2 - L^2$ this can be rewritten as  
$$
rR = \frac{1}{4}\Bigl(L(L-2d) - \frac{d^2 T}{L^2}\Bigr). \tag{31}
$$  
Now compute  
$$
\frac{rR L}{S(L-2d)} = \frac{ \frac{L}{4}\bigl(L(L-2d) - \frac{d^2 T}{L^2}\bigr) }{ \frac{L^3(L-2d)-d^2 T}{4L(L-2d)}\cdot(L-2d) } = \frac{ L\bigl(L(L-2d) - \frac{d^2 T}{L^2}\bigr) }{ \frac{L^3(L-2d)-d^2 T}{L} }.
$$  
Multiply numerator and denominator by $L$:  
$$
= \frac{ L^2\bigl(L(L-2d) - \frac{d^2 T}{L^2}\bigr) }{ L^3(L-2d)-d^2 T } = \frac{ L^3(L-2d) - d^2 T }{ L^3(L-2d)-d^2 T } = 1.
$$  

*Verification of (2) (the dot products):*  
$$
(A-M)\cdot\vec{v} = (-u)X + a(Y-a).
$$  
Using (5): $a(Y-a) = (u-r)(X+r)$, we get  
$$
(A-M)\cdot\vec{v} = -uX + (u-r)X + (u-r)r = -rX + r(u-r).
$$  
From the geometry, $u-r = X - \frac{L}{2}$. Hence $r(u-r) = rX - \frac{rL}{2}$. Substituting,  
$$
(A-M)\cdot\vec{v} = -rX + rX - \frac{rL}{2} = -\frac{rL}{2}.
$$  
For the second dot product, note that $(A-N) = (A-M) + (M-N)$. Since $M-N = (-d,0)$ and $(M-N)\cdot\vec{v} = -dX$, we have  
$$
(A-N)\cdot\vec{v} = -\frac{rL}{2} - dX.
$$  
But from the expressions for $r,R,X$ and $L$ one finds $dX = \frac{(R-r)L}{2}$. Hence  
$$
(A-N)\cdot\vec{v} = -\frac{rL}{2} - \frac{(R-r)L}{2} = -\frac{RL}{2}.
$$

*Verification of (6) (y-coordinate of $H$):*  
From (4) we have $y_H = \frac{(X-u)(v-X)}{Y}$. Using $X-u = d\bigl(\frac{X}{L}+\frac12\bigr)$ and $v-X = d\bigl(\frac12-\frac{X}{L}\bigr)$ (obtained from (8) and $v = u+d$), their product is $(X-u)(v-X) = d^2\bigl(\frac14 - \frac{X^2}{L^2}\bigr) = -\frac{d^2 T}{4L^2}$. Meanwhile from (13) we have $Y = \frac{d\,T}{4aL}$. Therefore  
$$
y_H = \frac{-\frac{d^2 T}{4L^2}}{\frac{dT}{4aL}} = -\frac{ad}{L}.
$$  

All steps are now fully justified. The proof is complete.