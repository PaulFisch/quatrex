# Pole-Subtracted Modal SCBA for Anharmonic Phonon NEGF

## A detailed formulation and implementation plan

### Status of this document

This document develops a proposed extension of the phonon–phonon SCBA/NEGF formulation in the attached project notes, especially the material corresponding to the report's Eqs. (1.26)–(1.29), (1.75), (1.78), (1.97), (1.110), (1.127)–(1.136), and the factorized-FC3 kernels of Sec. 1.6.

The **baseline NEGF and SCBA equations are taken from the report**. The pole subtraction, pole tracking, modal Schur-complement formulation, Keldysh pole-cluster treatment, and adaptive sector-selection strategy below are a **proposed algorithmic extension**. They should therefore be validated against the existing dense/FFT implementation before being used as a production approximation.

The central conclusion is:

> A decomposition into a singular/long-range part and a regular/short-range part is mathematically sound if it is formulated as an exact decomposition of the Green function and **all cross terms are retained**. Narrow isolated poles can be removed from the real-frequency grid and integrated analytically. Long-range propagating or weakly evanescent components can be retained in a low-dimensional modal representation rather than materialized as distant real-space blocks. The remaining Green function can then be both smoother in frequency and shorter-ranged in space.

There is, however, one important refinement to the simple statement
\[
G=G_{\mathrm{sing}}+G_{\mathrm{reg}}.
\]

For an open device the best conceptual decomposition is actually

\[
\boxed{
G^R
=
G_{\mathrm{pole}}^R
+
G_{\mathrm{cont}}^R
+
G_{\mathrm{reg}}^R
}
\tag{1}
\]

where

- \(G_{\mathrm{pole}}\) contains **isolated narrow frequency poles / quasiparticle resonances** that are suitable for residue calculus;
- \(G_{\mathrm{cont}}\) contains the **long-range propagating or weakly evanescent continuum contribution**, which is naturally represented by Bloch/complex-band modes but is not necessarily a finite set of frequency poles;
- \(G_{\mathrm{reg}}\) is the remaining **frequency-smooth and spatially decaying background**.

For the SCBA kernel these can be grouped into a binary representation

\[
G_{\mathrm{sing}}\equiv G_{\mathrm{pole}}+G_{\mathrm{cont}},
\qquad
G=G_{\mathrm{sing}}+G_{\mathrm{reg}},
\tag{2}
\]

but the implementation should remember that only \(G_{\mathrm{pole}}\) is automatically amenable to exact residue integration in frequency.

---

# 1. Why this decomposition is useful

The current solver has two related numerical difficulties.

First, narrow modes create frequency scales
\[
\Gamma_s\ll \omega_{\max},
\]
so a uniform real-frequency representation requires
\[
\Delta\omega\lesssim \Gamma_{\min}.
\tag{3}
\]

The report already derives this from the quasiparticle form
\[
G_s^R(\omega)
\simeq
\frac{1}
{\omega^2-\Omega_s^2+2i\Omega_s\Gamma_s}
\tag{4}
\]
near a positive-frequency resonance and explains why an unresolved peak makes the SCBA map stiff and grid-dependent.

Second, even when the Dyson matrix is block-banded, its inverse is not. Propagating components satisfy schematically
\[
G_{ij}\sim e^{iq(i-j)}
\tag{5}
\]
and do not decay with separation. Only evanescent components behave as
\[
G_{ij}\sim e^{-\kappa|i-j|}.
\tag{6}
\]

This is exactly the distinction emphasized in Sec. 1.5.5 of the report: a hard spatial band approximation is not an approximation to a generically banded Green function; it discards genuinely long-range propagating coherence.

The proposed method removes these two pathologies **before** the expensive SCBA convolution:

1. remove narrow poles from the numerical frequency representation;
2. remove slow spatial modes from the numerical real-space band;
3. evaluate these removed components analytically or in reduced mode space;
4. run the existing FFT/banded machinery only on a remainder that is broad in frequency and short-ranged in space.

The expected result is that the numerical resolution is determined by
\[
\Gamma_{\mathrm{reg}}
\gg
\Gamma_{\min}
\tag{7}
\]
and the spatial cutoff by the largest decay length left in \(G_{\mathrm{reg}}\), rather than by the sharpest/longest-ranged mode of the full Green function.

---

# 2. Baseline signed-frequency NEGF equations

For the pole analysis it is cleaner to use the signed complex frequency
\[
z\in\mathbb C
\]
rather than use \(\omega^2\) as the independent complex variable.

Define
\[
M^R(z)
=
z^2 I
-
D
-
\Sigma_L^R(z)
-
\Sigma_R^R(z)
-
\Sigma_s^R(z).
\tag{8}
\]

Then
\[
\boxed{
G^R(z)=M^R(z)^{-1}.
}
\tag{9}
\]

On the real axis,
\[
G^A(\omega)=G^R(\omega)^\dagger.
\tag{10}
\]

The Keldysh equations are
\[
\boxed{
G^{\lessgtr}(\omega)
=
G^R(\omega)
\Sigma_{\mathrm{tot}}^{\lessgtr}(\omega)
G^A(\omega)
}
\tag{11}
\]
with
\[
\Sigma_{\mathrm{tot}}
=
\Sigma_L+\Sigma_R+\Sigma_s.
\tag{12}
\]

For cubic anharmonicity, define the bilinear bubble map
\[
\begin{aligned}
\mathcal B(X,Y)_{\mu\mu'}(\omega)
&=
\frac{i\hbar}{2}
\sum_{abcd}
\Phi_{\mu ab}\Phi_{\mu'cd}
\\
&\quad\times
\int_{-\infty}^{\infty}
\frac{d\omega'}{2\pi}
X_{ad}(\omega')
Y_{bc}(\omega-\omega').
\end{aligned}
\tag{13}
\]

The SCBA self-energy is
\[
\boxed{
\Sigma_s^{\lessgtr}
=
\mathcal B(G^{\lessgtr},G^{\lessgtr}).
}
\tag{14}
\]

The retarded self-energy follows from
\[
\boxed{
\Sigma_s^R(\omega)
=
\frac12
\left[\Sigma_s^>(\omega)-\Sigma_s^<(\omega)\right]
+
i\mathcal P
\int\frac{d\omega'}{2\pi}
\frac{
\Sigma_s^>(\omega')-\Sigma_s^<(\omega')
}{
\omega-\omega'
}.
}
\tag{15}
\]

This is the starting point. The hybrid method must evaluate the **same functional** as closely as possible.

---

# 3. Exact local pole decomposition of \(G^R\)

## 3.1 Nonlinear eigenvalue problem

A pole \(z_\alpha\) satisfies
\[
M^R(z_\alpha)r_\alpha=0,
\tag{16}
\]
with left vector
\[
l_\alpha^\dagger M^R(z_\alpha)=0.
\tag{17}
\]

For a simple pole define
\[
d_\alpha
=
l_\alpha^\dagger
M^{R\prime}(z_\alpha)
r_\alpha,
\tag{18}
\]
where
\[
M^{R\prime}(z)
=
2zI
-
\partial_z\Sigma_L^R
-
\partial_z\Sigma_R^R
-
\partial_z\Sigma_s^R.
\tag{19}
\]

The residue is
\[
\boxed{
R_\alpha
=
\operatorname*{Res}_{z=z_\alpha}G^R(z)
=
\frac{
r_\alpha l_\alpha^\dagger
}{
d_\alpha
}.
}
\tag{20}
\]

This is the simple-eigenvalue form of the Keldysh theorem for nonlinear matrix pencils.

Therefore, inside an analytic region containing a selected set \(\mathcal P\) of simple poles,
\[
\boxed{
G^R(z)
=
\sum_{\alpha\in\mathcal P}
\frac{R_\alpha}{z-z_\alpha}
+
H^R(z),
}
\tag{21}
\]
where \(H^R(z)\) is analytic in that region.

Define
\[
G_{\mathrm{pole}}^R(z)
=
\sum_{\alpha\in\mathcal P}
\frac{R_\alpha}{z-z_\alpha},
\tag{22}
\]
and
\[
G_{\mathrm{bg}}^R(z)
=
G^R(z)-G_{\mathrm{pole}}^R(z).
\tag{23}
\]

Equation (23) is an **exact subtraction** for whatever poles are included in (22).

---

## 3.2 Bosonic pole pairing

For a real displacement problem, retarded bosonic symmetry produces positive- and negative-frequency partners. If
\[
z_{\alpha,+}
=
\Omega_\alpha-i\gamma_\alpha,
\qquad
\gamma_\alpha>0,
\tag{24}
\]
then the partner is
\[
z_{\alpha,-}
=
-z_{\alpha,+}^*
=
-\Omega_\alpha-i\gamma_\alpha.
\tag{25}
\]

The pole set used in the hybrid representation must be closed under this transformation.

Do not construct the negative-frequency pole by simply changing the sign of the real part while forgetting the residue transformation. The residue matrices must obey the corresponding bosonic transpose/conjugation relation inherited from
\[
G^R(-\omega)
\leftrightarrow
G^R(\omega)^*.
\tag{26}
\]

The exact transformation depends on the chosen \(q\)-space and index convention and should be tested against the same negative-frequency identity used by the current code,
\[
G^<_{ij}(\mathbf q,-\omega)
=
G^>_{ji}(-\mathbf q,\omega).
\tag{27}
\]

---

# 4. Important limitation: contacts produce branch cuts

A finite closed harmonic problem is meromorphic. An **open** device is not generally globally meromorphic because the lead surface Green functions contain propagating continua and band edges.

Thus a more accurate analytic statement is
\[
G^R
=
G_{\mathrm{isolated\ poles}}^R
+
G_{\mathrm{branch/continuum}}^R.
\tag{28}
\]

The residue representation (21) is exact only in a domain where
\[
M^R(z)
\]
is analytic except for the enclosed isolated poles.

This has an immediate implementation consequence:

> **Do not identify every propagating mode with a frequency pole.**

A propagating Bloch channel is a statement about spatial translation at a fixed real frequency. It need not correspond to an isolated physical-sheet pole of the open-device Green function.

Near a lead band edge the surface Green function has branch-point structure. Such a contribution should initially be put into the **continuum/modal sector**, not forced into a single-pole fit.

For in-band resonances one has two rigorous options:

1. implement the outgoing-wave analytic continuation of the contact self-energy to the appropriate resonance sheet and solve the nonlinear pole problem there;
2. keep the continuum on the real axis and use a validated local rational representation of its sharp resonance/background structure.

For a first implementation, option 2 is safer. True isolated poles in gaps or very weakly contact-coupled quasi-bound states can be handled first.

---

# 5. Exact convolution identities for the pole sector

The residue theorem itself is not the difficult part.

Consider
\[
I(a,b;\omega)
=
\int_{-\infty}^{\infty}
\frac{d\omega'}{2\pi}
\frac{1}{\omega'-a}
\frac{1}{\omega-\omega'-b}.
\tag{29}
\]

If
\[
\operatorname{Im}a<0,
\qquad
\operatorname{Im}b<0,
\]
then
\[
\boxed{
I(a,b;\omega)
=
-\frac{i}{\omega-a-b}.
}
\tag{30}
\]

Hence the convolution of two retarded poles is another pole whose location is the sum
\[
a+b.
\tag{31}
\]

With bosonic positive/negative partners this automatically produces
\[
\Omega_\alpha+\Omega_\beta
\tag{32}
\]
and
\[
\Omega_\alpha-\Omega_\beta
\tag{33}
\]
three-phonon structures.

---

## 5.1 Lorentzian form

Define
\[
L_{\Omega,\gamma}(\omega)
=
\frac{2\gamma}
{(\omega-\Omega)^2+\gamma^2},
\tag{34}
\]
so that
\[
\int\frac{d\omega}{2\pi}
L_{\Omega,\gamma}(\omega)=1.
\tag{35}
\]

Then
\[
\boxed{
L_{\Omega_1,\gamma_1}
*
L_{\Omega_2,\gamma_2}
=
L_{\Omega_1+\Omega_2,\gamma_1+\gamma_2}.
}
\tag{36}
\]

Thus two narrow quasiparticle peaks can be convolved analytically without resolving either peak on a uniform grid.

The zero-linewidth limit gives
\[
L_{\Omega,\gamma}
\rightarrow
2\pi\delta(\omega-\Omega),
\tag{37}
\]
and the usual golden-rule energy-conservation delta functions are recovered.

---

# 6. Re-analysis of the first major difficulty: pole tracking through SCBA

The statement "pole tracking is difficult" is correct, but it can be made much more precise.

The correct problem is **continuation of a nonlinear, non-Hermitian eigenvalue problem whose matrix function changes between SCBA iterations**.

This should not be implemented as "diagonalize \(G\) again and sort eigenvalues by frequency."

The robust design is:

1. global/contour initialization;
2. predictor from nonlinear-eigenvalue sensitivity;
3. local bordered-Newton correction;
4. cluster tracking rather than individual tracking near crossings;
5. periodic contour rescans as a recovery mechanism.

---

## 6.1 Predictor formula

Introduce a continuation parameter \(t\) between two SCBA iterates,
\[
M(z,t)r(t)=0.
\tag{38}
\]

Differentiate:
\[
M_z r\frac{dz}{dt}
+
M_t r
+
M\frac{dr}{dt}
=
0.
\tag{39}
\]

Left-multiplying by \(l^\dagger\) gives
\[
\boxed{
\frac{dz_\alpha}{dt}
=
-
\frac{
l_\alpha^\dagger M_t r_\alpha
}{
l_\alpha^\dagger M_z r_\alpha
}.
}
\tag{40}
\]

If only the scattering self-energy changes,
\[
M_t=-\partial_t\Sigma_s^R,
\]
so
\[
\boxed{
\frac{dz_\alpha}{dt}
=
\frac{
l_\alpha^\dagger
(\partial_t\Sigma_s^R)
r_\alpha
}{
d_\alpha
}.
}
\tag{41}
\]

For consecutive iterations,
\[
\Delta\Sigma_s^R
=
\Sigma_{s,n+1}^R-\Sigma_{s,n}^R,
\tag{42}
\]
use
\[
\boxed{
z_{\alpha,\mathrm{pred}}^{(n+1)}
=
z_\alpha^{(n)}
+
\frac{
l_\alpha^{(n)\dagger}
\Delta\Sigma_s^R(z_\alpha^{(n)})
r_\alpha^{(n)}
}{
d_\alpha^{(n)}
}.
}
\tag{43}
\]

This is a much better initial guess than restarting a global eigenvalue solve.

---

## 6.2 Bordered Newton corrector

Given a predicted pole, solve
\[
M(z)r=0
\tag{44}
\]
together with a gauge condition
\[
c^\dagger r=1.
\tag{45}
\]

One Newton step solves
\[
\boxed{
\begin{bmatrix}
M(z) & M'(z)r\\
c^\dagger & 0
\end{bmatrix}
\begin{bmatrix}
\delta r\\
\delta z
\end{bmatrix}
=
-
\begin{bmatrix}
M(z)r\\
c^\dagger r-1
\end{bmatrix}.
}
\tag{46}
\]

Then
\[
r\leftarrow r+\delta r,
\qquad
z\leftarrow z+\delta z.
\tag{47}
\]

After convergence, compute the left null vector
\[
l^\dagger M(z)=0
\tag{48}
\]
and normalize using
\[
l^\dagger M'(z)r=1.
\tag{49}
\]

With this normalization the residue is simply
\[
R_\alpha=r_\alpha l_\alpha^\dagger.
\tag{50}
\]

Recommended residual:
\[
\epsilon_{\mathrm{NEP}}
=
\frac{
\|M(z_\alpha)r_\alpha\|_2
}{
\left(
|z_\alpha|^2+\|D\|+\|\Sigma^R(z_\alpha)\|
\right)\|r_\alpha\|_2
}.
\tag{51}
\]

The pole should be accepted only if
\[
\epsilon_{\mathrm{NEP}}<\epsilon_{\mathrm{pole}}.
\tag{52}
\]

---

## 6.3 Pole conditioning

A simple nonlinear eigenvalue can be extremely sensitive when
\[
d_\alpha
=
l_\alpha^\dagger M'(z_\alpha)r_\alpha
\tag{53}
\]
is small.

A useful dimensionless condition indicator is
\[
\boxed{
\kappa_\alpha
=
\frac{
\|l_\alpha\|_2
\|r_\alpha\|_2
\|M'(z_\alpha)\|_2
}{
|l_\alpha^\dagger M'(z_\alpha)r_\alpha|
}.
}
\tag{54}
\]

Large \(\kappa_\alpha\) means:

- the residue may be very large;
- small SCBA changes can strongly move the pole;
- individual eigenvectors become a poor numerical object;
- the pole should probably be treated as part of a cluster.

Near an exceptional/defective point, the simple-pole expansion itself can fail and higher-order principal parts can appear. The implementation should **fall back to a cluster/rational representation** rather than attempt to maintain scalar pole labels.

---

# 7. Do not track individual modes through crossings: track subspaces

Suppose several poles satisfy
\[
|z_\alpha-z_\beta|
\lesssim
c_{\mathrm{cl}}
(\gamma_\alpha+\gamma_\beta).
\tag{55}
\]

They should be treated as one pole cluster
\[
\mathcal C.
\tag{56}
\]

For a cluster, individual eigenvectors can rotate arbitrarily under tiny perturbations even though the **invariant subspace is smooth**.

Let
\[
U_{\mathcal C}^{(n)}
\]
and
\[
U_{\mathcal C}^{(n+1)}
\]
be orthonormal bases of the right pole subspaces at two iterations.

The singular values of
\[
U_{\mathcal C}^{(n)\dagger}
U_{\mathcal C}^{(n+1)}
\tag{57}
\]
are the cosines of the principal angles.

Define
\[
\theta_{\max}
=
\max_j\theta_j.
\tag{58}
\]

A small \(\theta_{\max}\) means the same cluster has been tracked even if individual modes exchanged labels.

This is preferable to matching by frequency alone.

---

## 7.1 Contour/Riesz-style recovery

For a contour \(\Gamma\) lying in an analytic region, form moments
\[
\boxed{
A_k
=
\frac{1}{2\pi i}
\oint_\Gamma
z^k
M(z)^{-1}V\,dz,
}
\tag{59}
\]
with a small probing matrix \(V\).

For simple enclosed poles,
\[
A_k
=
\sum_{\alpha\in\Gamma}
z_\alpha^k
r_\alpha
l_\alpha^\dagger V.
\tag{60}
\]

Beyn-type contour methods extract the enclosed pole set and its subspace from these moments.

This should be used as:

- the initialization method;
- a periodic verification step;
- a recovery path when predictor/corrector fails;
- a pole-count check after large SCBA updates.

It is not necessary to run a global contour solve every iteration.

---

# 8. Recommended pole-tracking state machine

For every pole cluster store

```text
cluster_id
complex poles z_alpha
right vectors r_alpha
left vectors l_alpha
residues R_alpha
conditioning kappa_alpha
linewidths gamma_alpha
frequency window / contour
previous subspace basis
classification state
```

At SCBA iteration \(n+1\):

1. predict each cluster with Eq. (43);
2. correct each predicted pole with Eq. (46);
3. compute the new cluster subspace;
4. compare principal angles with the previous subspace;
5. reject a label match if the overlap is poor;
6. if a cluster loses/gains a pole, invoke a contour rescan;
7. every \(N_{\mathrm{rescan}}\) iterations, perform a contour audit even if tracking appears successful.

A practical matching cost between isolated candidates can combine pole displacement and vector overlap:
\[
C_{\alpha\beta}
=
w_z
\frac{
|z_\alpha^{(n)}-z_\beta^{(n+1)}|
}{
s_z
}
+
w_u
\left(
1-
|\hat r_\alpha^{(n)\dagger}\hat r_\beta^{(n+1)}|
\right).
\tag{61}
\]

Use a global assignment only for well-separated poles. For clusters, use subspace matching.

---

# 9. Branch cuts and resonance-sheet tracking

This deserves separate treatment because it is a larger issue than ordinary Newton convergence.

The contact self-energy
\[
\Sigma_c^R(z)
=
D_{CL}g_s^R(z)D_{LC}
\tag{62}
\]
inherits the analytic structure of the surface Green function.

On the real axis inside a lead band, propagating solutions have
\[
|\lambda|=1.
\tag{63}
\]

Near a band edge, propagating and evanescent branches coalesce and the decay length diverges. The corresponding frequency dependence is generally branch-point-like rather than a simple isolated pole.

Therefore:

### Safe initial pole set

Start by promoting modes that are

- in contact gaps;
- quasi-bound with extremely small contact broadening;
- isolated from band edges;
- numerically enclosed by a contour on which the chosen analytic continuation of \(M(z)\) is unambiguous.

### In-band resonances

For modes embedded in a continuum:

- either implement explicit outgoing-sheet continuation by continuously tracking the contact Bloch roots \(\lambda_\nu(z)\);
- or retain the continuum as \(G_{\mathrm{cont}}\) and fit only the local resonance/background response.

The second route is substantially easier to validate.

---

# 10. Re-analysis of the second major difficulty: nonequilibrium \(G^{<,>}\)

This is the part that needs the most care.

The earlier simplified statement
\[
G_\alpha^<
\simeq
W_\alpha^< L_\alpha
\tag{64}
\]
is valid only for an isolated pole with a sufficiently smooth projected source and negligible modal coherence.

The correct general construction starts from the **retarded pole decomposition**, not from an assumed scalar phonon occupation.

---

# 11. Exact Keldysh decomposition induced by the retarded split

Write
\[
G^R=P^R+B^R,
\tag{65}
\]
where
\[
P^R\equiv G_{\mathrm{pole}}^R
\tag{66}
\]
and
\[
B^R\equiv G^R-P^R.
\tag{67}
\]

Then
\[
G^A=P^A+B^A.
\tag{68}
\]

The exact Keldysh equation gives
\[
\begin{aligned}
G^{\lessgtr}
&=
(P^R+B^R)
\Sigma^{\lessgtr}
(P^A+B^A)
\\
&=
G_{PP}^{\lessgtr}
+
G_{PB}^{\lessgtr}
+
G_{BP}^{\lessgtr}
+
G_{BB}^{\lessgtr},
\end{aligned}
\tag{69}
\]
with
\[
G_{PP}^{\lessgtr}
=
P^R\Sigma^{\lessgtr}P^A,
\tag{70}
\]
\[
G_{PB}^{\lessgtr}
=
P^R\Sigma^{\lessgtr}B^A,
\tag{71}
\]
\[
G_{BP}^{\lessgtr}
=
B^R\Sigma^{\lessgtr}P^A,
\tag{72}
\]
\[
G_{BB}^{\lessgtr}
=
B^R\Sigma^{\lessgtr}B^A.
\tag{73}
\]

This identity is exact for the current SCBA iterate.

A particularly useful definition is
\[
\boxed{
G_{\mathrm{sing}}^{\lessgtr}
=
G_{PP}^{\lessgtr}
+
G_{PB}^{\lessgtr}
+
G_{BP}^{\lessgtr},
}
\tag{74}
\]
and
\[
\boxed{
G_{\mathrm{reg}}^{\lessgtr}
=
G_{BB}^{\lessgtr}.
}
\tag{75}
\]

Then **all terms containing at least one narrow retarded/advanced pole are removed from the ordinary numerical grid**.

This is more complete than defining the singular lesser Green function from \(PP\) alone.

---

# 12. Pole-cluster Keldysh matrix

Collect right and left vectors in
\[
U=[r_1,\ldots,r_{N_p}],
\qquad
V=[l_1,\ldots,l_{N_p}],
\tag{76}
\]
using the normalization
\[
l_\alpha^\dagger M'(z_\alpha)r_\alpha=1.
\tag{77}
\]

Then
\[
P^R(\omega)
=
U D^R(\omega)V^\dagger,
\tag{78}
\]
where
\[
D^R_{\alpha\beta}(\omega)
=
\delta_{\alpha\beta}
\frac{1}{\omega-z_\alpha}.
\tag{79}
\]

On the real axis,
\[
P^A(\omega)
=
V D^A(\omega)U^\dagger,
\tag{80}
\]
with
\[
D^A_{\alpha\alpha}
=
\frac{1}{\omega-z_\alpha^*}.
\tag{81}
\]

Define the projected Keldysh source
\[
\boxed{
S^{\lessgtr}(\omega)
=
V^\dagger
\Sigma_{\mathrm{tot}}^{\lessgtr}(\omega)
V.
}
\tag{82}
\]

Then
\[
\boxed{
G_{PP}^{\lessgtr}(\omega)
=
U
D^R(\omega)
S^{\lessgtr}(\omega)
D^A(\omega)
U^\dagger.
}
\tag{83}
\]

This is the correct low-dimensional nonequilibrium object.

---

# 13. Why a full modal density matrix is necessary

Equation (83) contains
\[
S_{\alpha\beta}^{\lessgtr},
\qquad
\alpha\neq\beta.
\tag{84}
\]

These are modal coherences.

For a cluster of overlapping resonances, the pole contribution is
\[
G_{PP}^{\lessgtr}
=
\sum_{\alpha\beta}
r_\alpha
\frac{
S_{\alpha\beta}^{\lessgtr}(\omega)
}{
(\omega-z_\alpha)(\omega-z_\beta^*)
}
r_\beta^\dagger.
\tag{85}
\]

Replacing this by independent scalar occupations
\[
n_\alpha
\tag{86}
\]
throws away the terms \(\alpha\neq\beta\).

That approximation is only justified when both of the following hold:

1. the poles are well separated compared with their widths;
2. the projected source is nearly diagonal.

A useful coherence metric is
\[
\boxed{
\epsilon_{\mathrm{coh}}
=
\frac{
\|\operatorname{offdiag}N_{\mathcal C}\|_F
}{
\|N_{\mathcal C}\|_F
},
}
\tag{87}
\]
where
\[
N_{\mathcal C}
=
\frac{i}{2\pi}
\int_{\mathcal W_{\mathcal C}}
d\omega\,
D^R S^< D^A
\tag{88}
\]
is the pole-cluster covariance/occupation matrix.

Only if
\[
\epsilon_{\mathrm{coh}}\ll 1
\tag{89}
\]
should the cluster be reduced to independent scalar modal populations.

Under a temperature bias, keeping the matrix form by default is safer.

---

# 14. Smooth-source approximation

For an isolated narrow pole,
\[
\gamma_\alpha
\ll
\Delta_{\mathrm{src}},
\tag{90}
\]
where \(\Delta_{\mathrm{src}}\) is the scale on which
\[
S^{\lessgtr}(\omega)
\]
varies, one may approximate
\[
S^{\lessgtr}(\omega)
\simeq
S^{\lessgtr}(\Omega_\alpha)
\tag{91}
\]
inside the pole window.

For a diagonal isolated pole,
\[
G_{PP,\alpha}^{\lessgtr}(\omega)
\simeq
r_\alpha r_\alpha^\dagger
\frac{
S_{\alpha\alpha}^{\lessgtr}(\Omega_\alpha)
}{
(\omega-\Omega_\alpha)^2+\gamma_\alpha^2
}.
\tag{92}
\]

The integrated weight is
\[
\int\frac{d\omega}{2\pi}
\frac{1}{
(\omega-\Omega)^2+\gamma^2
}
=
\frac{1}{2\gamma}.
\tag{93}
\]

Thus the scalar weight is proportional to
\[
\frac{
S_{\alpha\alpha}^{\lessgtr}(\Omega_\alpha)
}{
2\gamma_\alpha
}.
\tag{94}
\]

This is the controlled origin of the Lorentzian pole weight.

---

# 15. What if the projected source is not smooth?

This occurs especially

- near \(\omega=0\);
- near a lead band edge;
- if another narrow resonance overlaps the pole window;
- under strong nonequilibrium where \(\Sigma_s^{<,>}\) itself develops sharp structure.

Do **not** freeze \(S^{<,>}\) in that case.

Instead, because \(S^{<,>}\) is only an \(N_p\times N_p\) matrix, use one of the following:

### Option A — local adaptive quadrature

Evaluate the small matrix
\[
D^R(\omega)S^{\lessgtr}(\omega)D^A(\omega)
\tag{95}
\]
with pole-aware adaptive quadrature.

This may require many scalar frequency evaluations but does **not** require a huge matrix-valued device grid.

### Option B — local polynomial moments

Approximate
\[
S^{\lessgtr}(\omega)
\simeq
\sum_{m=0}^{p}
S_m(\omega-\Omega_c)^m
\tag{96}
\]
within a pole cluster.

The required integrals are then moments of known rational functions and can be evaluated analytically.

### Option C — local rational approximation

Fit
\[
S^{\lessgtr}(\omega)
\simeq
S_\infty
+
\sum_j
\frac{C_j}{\omega-p_j}.
\tag{97}
\]

Then every pole-sector convolution becomes a finite residue sum.

This is likely the highest-performance final implementation because the fitting problem is small-dimensional.

---

# 16. Positivity and Keldysh consistency

The advantage of Eq. (83) is that it is a congruence of the projected source:
\[
G_{PP}^{\lessgtr}
=
(U D^R)
S^{\lessgtr}
(U D^R)^\dagger.
\tag{98}
\]

Therefore, if the chosen sign convention makes the source semidefinite, the pole-pole sector inherits that semidefiniteness automatically.

This is safer than assigning modal weights independently and clipping negative occupations afterward.

Recommended tests at every iteration:

\[
G^R-G^A
=
G^>-G^<,
\tag{99}
\]

\[
\Sigma^R-\Sigma^A
=
\Sigma^>-\Sigma^<,
\tag{100}
\]

and at equilibrium
\[
\Sigma^>(\omega)
=
e^{\beta\hbar\omega}
\Sigma^<(\omega).
\tag{101}
\]

The bosonic negative-frequency relation must also hold separately for the reconstructed pole+background result.

---

# 17. Mixed Keldysh pole/background terms

The mixed pieces are
\[
G_{PB}^{\lessgtr}
=
U D^R
\underbrace{
V^\dagger
\Sigma^{\lessgtr}
B^A
}_{K_{PB}^{\lessgtr}(\omega)},
\tag{102}
\]
and
\[
G_{BP}^{\lessgtr}
=
\underbrace{
B^R
\Sigma^{\lessgtr}
V
}_{K_{BP}^{\lessgtr}(\omega)}
D^A U^\dagger.
\tag{103}
\]

The matrices
\[
K_{PB}^{\lessgtr},
\quad
K_{BP}^{\lessgtr}
\tag{104}
\]
are smooth if the background really is regular.

This suggests an efficient representation:

- store the pole denominator analytically;
- evaluate only the projected coefficient \(K(\omega)\) on a coarse grid;
- interpolate or rationally fit \(K(\omega)\).

This removes the narrow denominator from the grid while retaining the pole-background interference.

The latter is important: resonance/background interference is a real part of scattering Green-function decompositions and should not be dropped by default.

---

# 18. SCBA decomposition after the Keldysh split

Once
\[
G^{\lessgtr}
=
G_S^{\lessgtr}
+
G_R^{\lessgtr},
\tag{105}
\]
the bubble is exactly

\[
\boxed{
\Sigma_s^{\lessgtr}
=
\Sigma_{SS}^{\lessgtr}
+
\Sigma_{SR}^{\lessgtr}
+
\Sigma_{RS}^{\lessgtr}
+
\Sigma_{RR}^{\lessgtr},
}
\tag{106}
\]
with
\[
\Sigma_{SS}^{\lessgtr}
=
\mathcal B(G_S^{\lessgtr},G_S^{\lessgtr}),
\tag{107}
\]
\[
\Sigma_{SR}^{\lessgtr}
=
\mathcal B(G_S^{\lessgtr},G_R^{\lessgtr}),
\tag{108}
\]
\[
\Sigma_{RS}^{\lessgtr}
=
\mathcal B(G_R^{\lessgtr},G_S^{\lessgtr}),
\tag{109}
\]
\[
\Sigma_{RR}^{\lessgtr}
=
\mathcal B(G_R^{\lessgtr},G_R^{\lessgtr}).
\tag{110}
\]

No propagating/narrow mode is projected out of the physics.

The decomposition changes the **representation**, not the diagram.

---

# 19. Pole-pole bubble in reduced mode space

For the simple diagonal form
\[
G_S^{\lessgtr}(\omega)
=
\sum_\alpha
w_\alpha^{\lessgtr}(\omega)
u_\alpha u_\alpha^\dagger,
\tag{111}
\]
define
\[
\boxed{
V_\mu^{\alpha\beta}
=
\sum_{ab}
\Phi_{\mu ab}
u_{\alpha a}
u_{\beta b}.
}
\tag{112}
\]

Then
\[
\boxed{
\Sigma_{SS,\mu\mu'}^{\lessgtr}(\omega)
=
\frac{i\hbar}{2}
\sum_{\alpha\beta}
C_{\alpha\beta}^{\lessgtr}(\omega)
V_\mu^{\alpha\beta}
V_{\mu'}^{\alpha\beta *},
}
\tag{113}
\]
where
\[
C_{\alpha\beta}^{\lessgtr}
=
w_\alpha^{\lessgtr}
*
w_\beta^{\lessgtr}.
\tag{114}
\]

If the weights are Lorentzians, Eq. (114) is analytic.

The output is itself a sum of low-rank outer products.

---

# 20. General coherent pole-cluster bubble

For a cluster
\[
G_S^{\lessgtr}
=
U F^{\lessgtr}U^\dagger,
\tag{115}
\]
define
\[
\bar\Phi_{\mu,\alpha\beta}
=
\sum_{ab}
\Phi_{\mu ab}
U_{a\alpha}
U_{b\beta}.
\tag{116}
\]

Then
\[
\begin{aligned}
\Sigma_{SS,\mu\mu'}^{\lessgtr}
&=
\frac{i\hbar}{2}
\sum_{\alpha\beta\gamma\delta}
\bar\Phi_{\mu,\alpha\beta}
\bar\Phi_{\mu',\gamma\delta}^*
\\
&\quad\times
\int\frac{d\omega'}{2\pi}
F_{\alpha\delta}^{\lessgtr}(\omega')
F_{\beta\gamma}^{\lessgtr}(\omega-\omega').
\end{aligned}
\tag{117}
\]

All large device indices have been removed from the convolution.

This is the correct formula when modal coherences matter.

---

# 21. Mixed self-energy term after one modal projection

For
\[
G_S^{\lessgtr}
=
\sum_\alpha
w_\alpha^{\lessgtr}
u_\alpha u_\alpha^\dagger,
\tag{118}
\]
define
\[
[B_\alpha]_{\mu b}
=
\sum_a
\Phi_{\mu ab}
u_{\alpha a}.
\tag{119}
\]

Then
\[
\boxed{
\Sigma_{SR}^{\lessgtr}(\omega)
=
\frac{i\hbar}{2}
\sum_\alpha
\int\frac{d\omega'}{2\pi}
w_\alpha^{\lessgtr}(\omega')
B_\alpha
G_R^{\lessgtr}(\omega-\omega')
B_\alpha^\dagger.
}
\tag{120}
\]

Define
\[
K_\alpha^{\lessgtr}(\omega)
=
B_\alpha
G_R^{\lessgtr}(\omega)
B_\alpha^\dagger.
\tag{121}
\]

Then
\[
\boxed{
\Sigma_{SR}^{\lessgtr}
=
\frac{i\hbar}{2}
\sum_\alpha
w_\alpha^{\lessgtr}*K_\alpha^{\lessgtr}.
}
\tag{122}
\]

This is much cheaper than a full four-index ring.

If \(K_\alpha\) is smooth, the pole kernel can be handled analytically and \(K_\alpha\) only needs a coarse representation.

---

# 22. Re-analysis of the third difficulty: deciding what belongs in each sector

This should not be done by a single heuristic such as

> "if \(|\lambda|=1\), make it singular."

There are **four separate questions**:

1. Is it too narrow for the regular frequency grid?
2. Is it too long-ranged for the regular spatial band?
3. Is the mode isolated enough to admit a stable pole representation?
4. Is it important enough to the FC3 bubble to justify special treatment?

A mode can be:

- narrow but spatially localized;
- broad but spatially propagating;
- both narrow and propagating;
- neither.

Therefore frequency and spatial classification should be independent.

---

# 23. Frequency promotion criterion

Let the regular grid spacing be
\[
h_{\mathrm{reg}}.
\tag{123}
\]

If the desired number of samples per half-width is \(p_\Gamma\), a pole is under-resolved when
\[
h_{\mathrm{reg}}
>
\frac{\gamma_\alpha}{p_\Gamma}.
\tag{124}
\]

Define
\[
\boxed{
q_{\omega,\alpha}
=
\frac{
\gamma_\alpha
}{
p_\Gamma h_{\mathrm{reg}}
}.
}
\tag{125}
\]

Promote to the pole sector when
\[
q_{\omega,\alpha}<1.
\tag{126}
\]

Use hysteresis:

- enter pole sector when \(q_\omega<q_{\mathrm{in}}\);
- leave pole sector only when \(q_\omega>q_{\mathrm{out}}\);
- choose
\[
q_{\mathrm{out}}>q_{\mathrm{in}}.
\tag{127}
\]

This prevents sector thrashing as the linewidth changes during SCBA.

---

# 24. Spatial promotion criterion from complex bands

For a bulk/reference mode with Bloch factor
\[
\lambda_\nu,
\tag{128}
\]
the amplitude after \(n\) principal layers is
\[
|\lambda_\nu|^n.
\tag{129}
\]

For an evanescent mode define
\[
\boxed{
\xi_\nu
=
-\frac{1}{\ln|\lambda_\nu|}
}
\tag{130}
\]
in principal-layer units.

If all modes retained in the regular sector obey
\[
|\lambda_\nu|\le\rho<1,
\tag{131}
\]
then
\[
\|G_{R,i,i+n}\|
\lesssim
C\rho^n.
\tag{132}
\]

The neglected tail after band \(b\) satisfies
\[
\boxed{
E_x(b)
\lesssim
C\frac{\rho^{b+1}}{1-\rho}.
}
\tag{133}
\]

Therefore choose \(b\) from
\[
C\frac{\rho^{b+1}}{1-\rho}
<
\epsilon_x.
\tag{134}
\]

Equivalently, if a mode violates this bound for the allowed maximum band \(b_{\max}\), promote that mode to the spatial modal sector.

Every genuinely propagating mode with
\[
|\lambda|=1
\tag{135}
\]
is formally infinite-ranged and belongs in the modal sector unless its coupling/residue is numerically negligible.

---

# 25. Isolation criterion

Define
\[
\boxed{
\eta_\alpha
=
\min_{\beta\neq\alpha}
\frac{
|z_\alpha-z_\beta|
}{
\gamma_\alpha+\gamma_\beta
}.
}
\tag{136}
\]

If
\[
\eta_\alpha>\eta_{\mathrm{iso}},
\tag{137}
\]
the pole can be treated independently.

If
\[
\eta_\alpha\lesssim\eta_{\mathrm{iso}},
\tag{138}
\]
promote the entire group to a coherent cluster.

Do not force a diagonal scalar-mode ansatz onto such a cluster.

---

# 26. Band-edge / branch-cut criterion

Let
\[
\Delta_{\mathrm{edge},\alpha}
\]
be the distance from the pole frequency to the nearest contact band edge or known branch point.

If
\[
\Delta_{\mathrm{edge},\alpha}
\lesssim
c_{\mathrm{edge}}\gamma_\alpha,
\tag{139}
\]
do not use a simple isolated-pole approximation.

Assign it to

- the continuum/modal sector, or
- a local rational cluster containing the branch-edge structure.

This is particularly important near van Hove singularities.

---

# 27. Importance criterion

A pole with tiny residue may be mathematically narrow but physically irrelevant.

A first spectral-weight indicator is
\[
W_\alpha
=
\int_{\mathcal W_\alpha}
\frac{d\omega}{2\pi}
\|A_\alpha(\omega)\|_F.
\tag{140}
\]

A more targeted SCBA indicator includes the cubic vertex:
\[
\boxed{
I_\alpha
=
\left[
\sum_\beta
\|V^{\alpha\beta}\|_F^2
W_\alpha W_\beta
\right]^{1/2}.
}
\tag{141}
\]

A mode with small \(I_\alpha\) contributes little to the bubble.

However, an unresolved pole should only be **dropped entirely** if both

- its spectral weight is negligible;
- its FC3-weighted contribution is negligible.

Otherwise it should be promoted, not ignored.

---

# 28. Recommended dynamic classification logic

For each candidate mode/cluster evaluate:

```text
frequency_score
spatial_decay_score
isolation_score
pole_condition_number
distance_to_contact_branch_edge
spectral_weight
FC3_weighted_importance
```

Then:

### Pole sector

Use if

```text
unresolved in frequency
AND analytically isolated
AND pole condition acceptable
AND not too close to a branch point
```

### Continuum/modal sector

Use if

```text
long-ranged in space
BUT not representable as a stable isolated frequency pole
```

This includes propagating channels and band-edge modes.

### Regular sector

Use only if

```text
resolved on coarse frequency representation
AND sufficiently short-ranged for selected spatial band
```

This is the crucial condition. The regular sector should be defined by what the numerical solver can actually represent accurately.

---

# 29. Do not change the decomposition every SCBA iteration

Even if the exact algebra is invariant under repartitioning, an approximate implementation is not.

If a mode jumps between sectors every iteration, the numerical fixed-point map itself changes discontinuously.

Use **adaptation epochs**:

1. choose a sector partition;
2. hold it fixed for \(N_{\mathrm{epoch}}\) SCBA steps;
3. converge substantially within that representation;
4. evaluate promotion/demotion indicators;
5. change the partition only if a hysteresis threshold is crossed;
6. migrate the state conservatively;
7. continue.

A particularly safe strategy is:

- freeze the partition during the final Newton/Anderson convergence phase;
- only reclassify between outer restarts.

---

# 30. The spatial mode problem: do not diagonalize \(G(\omega)\) pointwise

A pointwise eigendecomposition
\[
G^R(\omega)
=
X(\omega)\Lambda_G(\omega)X(\omega)^{-1}
\tag{142}
\]
does **not** in general separate propagating from evanescent waves.

Propagating/evanescent character is defined by the translation operator.

For nearest-neighbor principal layers the bulk mode equation is
\[
\boxed{
\left[
-H_{10}\lambda^{-1}
+
(z^2I-H_{00})
-
H_{01}\lambda
\right]v
=
0.
}
\tag{143}
\]

With an effective periodic scattering self-energy, replace
\[
H_{00}
\rightarrow
H_{00}+\Sigma_{00}^R,
\quad
H_{01}
\rightarrow
H_{01}+\Sigma_{01}^R
\tag{144}
\]
as appropriate.

The eigenvalue
\[
\lambda=e^{iq}
\tag{145}
\]
determines spatial behavior.

This is the correct modal decomposition.

---

# 31. What changes under anharmonic SCBA?

In equilibrium in a homogeneous structure, the self-consistent self-energy may retain translational periodicity, so an effective complex-band problem remains meaningful.

Under a thermal bias,
\[
\Sigma_s^R(i,\omega)
\]
is generally position dependent.

Therefore harmonic plane waves are **not exact eigenmodes of the full Dyson operator**.

The safe formulation is not to freeze the first-iteration modes as independent channels. Instead use them as a **reduced basis** coupled through a Schur complement.

---

# 32. Exact modal/regular Schur-complement formulation

Let \(U\) span the selected long-range right modal subspace and \(W\) be a dual basis satisfying
\[
W^\dagger U=I.
\tag{146}
\]

Define the oblique projector
\[
P=UW^\dagger,
\qquad
Q=I-P.
\tag{147}
\]

In the \(P/Q\) decomposition,
\[
M
=
\begin{bmatrix}
M_{PP} & M_{PQ}\\
M_{QP} & M_{QQ}
\end{bmatrix}.
\tag{148}
\]

The exact \(P\)-space Green function is
\[
\boxed{
G_{PP}
=
\left[
M_{PP}
-
M_{PQ}M_{QQ}^{-1}M_{QP}
\right]^{-1}.
}
\tag{149}
\]

Define the Schur complement
\[
S_P
=
M_{PP}
-
M_{PQ}M_{QQ}^{-1}M_{QP}.
\tag{150}
\]

Then
\[
G_{PP}=S_P^{-1}.
\tag{151}
\]

The cross blocks are
\[
G_{PQ}
=
-
G_{PP}M_{PQ}M_{QQ}^{-1},
\tag{152}
\]
\[
G_{QP}
=
-
M_{QQ}^{-1}M_{QP}G_{PP},
\tag{153}
\]
and
\[
G_{QQ}
=
M_{QQ}^{-1}
+
M_{QQ}^{-1}M_{QP}G_{PP}M_{PQ}M_{QQ}^{-1}.
\tag{154}
\]

This formulation has exactly the desired structure:

- the long-range sector is small-dimensional but globally coupled;
- the regular \(Q\) inverse can be computed with a banded/local method;
- coupling between propagating and evanescent sectors is **not discarded**.

This is the mathematically clean version of "solve the plane waves analytically and the decaying part numerically."

---

# 33. Pole extraction can be performed on the reduced Schur complement

An additional saving follows immediately.

Instead of finding poles from
\[
\det M(z)=0
\tag{155}
\]
in the full device space, solve
\[
\boxed{
\det S_P(z)=0
}
\tag{156}
\]
for poles dominated by the selected modal subspace.

Since
\[
S_P
\]
has dimension equal to the number of retained long-range modes, pole tracking can become much cheaper.

The effect of the regular sector enters exactly through
\[
M_{PQ}M_{QQ}^{-1}M_{QP}.
\tag{157}
\]

This is preferable to simply removing \(Q\)-space physics.

---

# 34. Factorized spatial Green function

For a homogeneous or reduced modal region, write
\[
\boxed{
G_{S,ij}^R(\omega)
=
U_i(\omega)
C^R(\omega)
V_j^\dagger(\omega).
}
\tag{158}
\]

The matrices

- \(U_i\): right modal amplitudes at slab \(i\);
- \(V_j\): dual modal amplitudes at slab \(j\);
- \(C\): small modal Green matrix;

replace the full dense family \(G_{ij}\).

For a pure bulk mode,
\[
U_i
\sim
v_\alpha\lambda_\alpha^i.
\tag{159}
\]

A finite spatial sum is analytic:
\[
\boxed{
\sum_{n=n_0}^{N-1}\lambda^n
=
\lambda^{n_0}
\frac{
1-\lambda^{N-n_0}
}{
1-\lambda
}.
}
\tag{160}
\]

For
\[
\lambda\rightarrow1,
\]
use the stable limit
\[
\sum_{n=n_0}^{N-1}\lambda^n
\rightarrow
N-n_0.
\tag{161}
\]

Thus long-range plane-wave factors do not need to be materialized block by block merely to be summed.

---

# 35. Combination with the report's factorized FC3

Suppose the cubic tensor is represented as
\[
\boxed{
\Phi_{\mu ab}
\simeq
\sum_{r=1}^{R_\Phi}
\lambda_r
A_{\mu r}
B_{ar}
C_{br}.
}
\tag{162}
\]

For a singular mode \(u_\alpha\), define
\[
s_{r\alpha}
=
\sum_a
B_{ar}u_{\alpha a},
\tag{163}
\]
\[
t_{r\beta}
=
\sum_b
C_{br}u_{\beta b}.
\tag{164}
\]

Then
\[
\boxed{
V^{\alpha\beta}
=
A
\left[
\lambda
\odot
s_\alpha
\odot
t_\beta
\right].
}
\tag{165}
\]

Thus the pole-pole cubic projection no longer requires a three-index contraction at runtime.

For the mixed term,
\[
B_\alpha
=
A D_\alpha C^T,
\tag{166}
\]
where
\[
D_\alpha
=
\operatorname{diag}
(\lambda_r s_{r\alpha}).
\tag{167}
\]

Hence
\[
\boxed{
B_\alpha
G_R
B_\alpha^\dagger
=
A D_\alpha
\left(
C^T G_R C^*
\right)
D_\alpha^\dagger
A^\dagger.
}
\tag{168}
\]

The large regular Green function enters only through the small Gram matrix
\[
\boxed{
\mathcal G_R
=
C^T G_R C^*.
}
\tag{169}
\]

This is likely the most important performance synergy in the proposal:

\[
\boxed{
\text{factorized FC3}
+
\text{factorized }G_{\mathrm{sing}}
+
\text{banded }G_{\mathrm{reg}}.
}
\tag{170}
\]

---

# 36. Retarded self-energy: subtract analytic pieces before Hilbert transform

Let
\[
\Delta\Sigma
=
\Sigma^>-\Sigma^<.
\tag{171}
\]

Decompose
\[
\Delta\Sigma
=
\Delta\Sigma_{\mathrm{analytic}}
+
\Delta\Sigma_{\mathrm{reg}}.
\tag{172}
\]

If a contribution has an explicit retarded pole representation
\[
\Sigma_{\mathrm{analytic}}^R(z)
=
\sum_j
\frac{C_j}{z-p_j},
\qquad
\operatorname{Im}p_j<0,
\tag{173}
\]
its real and imaginary parts already satisfy Kramers–Kronig analytically.

Do not numerically Hilbert-transform that contribution.

Use
\[
\boxed{
\Sigma_s^R
=
\Sigma_{\mathrm{analytic}}^R
+
\frac12\Delta\Sigma_{\mathrm{reg}}
+
i\mathcal H[\Delta\Sigma_{\mathrm{reg}}].
}
\tag{174}
\]

This makes the numerical principal-value problem smoother and reduces sensitivity to the high-resolution pole region.

---

# 37. Conservation: what must remain exact

The original cubic SCBA bubble is conserving when it is constructed self-consistently from the same dressed \(G\) on both internal lines.

An algebraic decomposition
\[
G=G_S+G_R
\tag{175}
\]
does not change that because
\[
\mathcal B(G,G)
=
\mathcal B(S,S)
+
\mathcal B(S,R)
+
\mathcal B(R,S)
+
\mathcal B(R,R).
\tag{176}
\]

Problems begin only when different uncontrolled approximations are applied to different sectors.

The implementation should therefore enforce:

### Same reconstructed \(G\) on both bubble legs

Do not use a pole model on one leg and a different approximation on the other.

### Symmetric cross terms

Evaluate
\[
SR+RS
\tag{177}
\]
with mutually consistent quadrature/interpolation.

### Adjoint grid transfer

If regular quantities are transferred between a coarse grid and an auxiliary convolution grid, use the same energy-weighted adjoint principle already developed in the report's dual-grid Eq. (1.111), rather than pointwise sampling.

### Preserve cubic-vertex permutation symmetry

The FC3 used in every sector must be the same symmetric tensor/factorization.

---

# 38. Conservation diagnostics

At every accepted SCBA iterate measure

\[
\boxed{
J_s
=
\int
\frac{d\omega}{2\pi}
\hbar\omega
\operatorname{Tr}
\left[
\Sigma_s^>G^<
-
\Sigma_s^<G^>
\right].
}
\tag{178}
\]

At a conserving fixed point,
\[
J_s\rightarrow0.
\tag{179}
\]

Also check
\[
J_L+J_R\rightarrow0,
\tag{180}
\]
and the slab-resolved telescoped current from Sec. 1.4.5 of the report.

The individual terms
\[
J_{SS},J_{SR},J_{RS},J_{RR}
\tag{181}
\]
need not vanish separately. Only their sum is constrained.

This sector-resolved breakdown is nevertheless an excellent debugging tool.

---

# 39. Causality and semidefiniteness

The report shows that a sharp real-space boxcar truncation can make the bubble noncausal because the mask itself is indefinite.

The hybrid method should avoid reproducing that problem in another form.

Recommended rules:

1. never truncate individual residue matrix elements arbitrarily;
2. truncate by complete modes/subspaces;
3. keep a biorthogonal pair \(r_\alpha,l_\alpha\) together;
4. if a regular real-space mask is still needed, use the causal/PSD-compatible strategy already derived in the report;
5. reconstruct \(\Sigma^R\) from a causal analytic pole part plus a Kramers–Kronig-consistent regular part.

---

# 40. Acoustic zero modes

The exact acoustic origin should initially be excluded from the ordinary simple-pole machinery.

Near
\[
\omega=0
\]
the assumptions
\[
\gamma\ll\Omega
\tag{182}
\]
and isolated Lorentzian quasiparticle behavior are not appropriate.

The Bose distribution also has the singular expansion
\[
n_B(\omega)
\sim
\frac{k_BT}{\hbar\omega}.
\tag{183}
\]

Recommended first implementation:

- retain the report's dedicated \(\omega=0\) convention;
- keep a small low-frequency window as a separate continuum sector;
- apply pole subtraction only above
\[
\omega_{\mathrm{pole,min}}>0.
\tag{184}
\]

A later acoustic treatment can use analytic long-wavelength dispersions rather than a generic isolated-pole model.

---

# 41. Van Hove singularities and band edges

A van Hove/band-edge feature may be sharp in frequency but is not generically a simple Lorentzian pole.

Near such points:

- the complex-band decay length diverges;
- several Bloch roots can coalesce;
- the contact surface Green function can have square-root-type branch behavior.

Therefore use

\[
G_{\mathrm{cont}}
\]
rather than
\[
G_{\mathrm{pole}}
\]
for these features.

If desired, approximate the continuum locally by a multi-pole rational representation, but validate the approximation against direct real-axis samples.

---

# 42. Error estimators

A production implementation needs explicit error indicators.

## 42.1 Pole reconstruction error

At audit frequencies \(\omega_j\),
\[
\boxed{
\epsilon_G(\omega_j)
=
\frac{
\|G_{\mathrm{direct}}^R(\omega_j)
-
G_{\mathrm{pole}}^R(\omega_j)
-
G_{\mathrm{bg}}^R(\omega_j)\|
}{
\|G_{\mathrm{direct}}^R(\omega_j)\|
}.
}
\tag{185}
\]

The background is not useful if this subtraction is inaccurate.

---

## 42.2 Pole-window Keldysh error

Compare
\[
G_{\mathrm{direct}}^{\lessgtr}
\]
against the reconstructed pole+background Keldysh form on a small dense audit grid around each pole.

Use
\[
\epsilon_K
=
\max_{\omega\in\mathcal W_\alpha}
\frac{
\|G_{\mathrm{direct}}^{\lessgtr}
-
G_{\mathrm{hybrid}}^{\lessgtr}\|
}{
\|G_{\mathrm{direct}}^{\lessgtr}\|+\epsilon_0
}.
\tag{186}
\]

---

## 42.3 Spatial tail error

Use Eq. (133):
\[
\epsilon_x
\sim
C\frac{\rho^{b+1}}{1-\rho}.
\tag{187}
\]

Also compare selected distant blocks from the full recursion against the modal+regular representation.

---

## 42.4 Self-energy audit

On a small system or sparse set of frequencies compute both
\[
\Sigma_{\mathrm{dense}}
\]
and
\[
\Sigma_{\mathrm{hybrid}}.
\]

Track
\[
\boxed{
\epsilon_\Sigma
=
\frac{
\|\Sigma_{\mathrm{hybrid}}-\Sigma_{\mathrm{dense}}\|_F
}{
\|\Sigma_{\mathrm{dense}}\|_F
}.
}
\tag{188}
\]

---

# 43. Data structures

A possible implementation layout is:

## Pole cluster

```text
PoleCluster:
    ids
    z[Np]
    R_right[Ndof, Np]
    R_left[Ndof, Np]
    residue_norm[Np]
    condition[Np]
    gamma[Np]
    frequency_window
    contour
    subspace_Q
    source_less[Np, Np, ...]
    source_greater[Np, Np, ...]
    classification_flags
```

The full device vectors should preferably be stored in block/modal factorized form if spatial factorization is active.

## Regular Green function

```text
RegularGF:
    coarse_frequency_grid
    selected_block_band
    G_R
    G_less
    G_greater
    projected_Grams_for_FC3
```

## Continuum modal sector

```text
ContinuumModes:
    frequency_grid
    lambda[mode, frequency]
    right_mode_vectors
    dual_mode_vectors
    group_velocity_or_outgoing_flag
    decay_length
    modal_coupling_matrix
```

## Hybrid self-energy

```text
HybridSigma:
    analytic_pole_terms
    mixed_projected_terms
    regular_grid_terms
    retarded_analytic_terms
    retarded_regular_terms
```

---

# 44. Proposed SCBA iteration

At iteration \(n\):

## Step 1 — construct current Dyson operator

\[
M_n^R(z)
=
z^2I
-
D
-
\Sigma_c^R(z)
-
\Sigma_{s,n}^R(z).
\tag{189}
\]

## Step 2 — update pole clusters

- predict with Eq. (43);
- correct with Eq. (46);
- update residues;
- update conditioning;
- cluster nearby modes;
- contour-rescan failed clusters.

## Step 3 — update spatial modal sector

At the regular frequency samples:

- solve/update complex-band modes;
- retain propagating and slow evanescent channels;
- track mode subspaces by overlap/principal angles;
- build \(U,W\);
- form/update the reduced Schur complement if spatial reduction is enabled.

## Step 4 — build \(G^R\)

Represent
\[
G^R
=
G_{\mathrm{pole}}^R
+
G_{\mathrm{cont}}^R
+
G_{\mathrm{reg}}^R.
\tag{190}
\]

Do not materialize distant long-range blocks if they can be generated from modal factors.

## Step 5 — construct \(G^{<,>}\)

Use the exact Keldysh decomposition of Eqs. (69)–(75).

For each pole cluster:

- compute the projected source \(S^{<,>}\);
- keep the full coherence matrix by default;
- use local rational/polynomial representation if it varies over the pole window.

## Step 6 — evaluate the bubble

Compute
\[
SS,\quad SR,\quad RS,\quad RR.
\tag{191}
\]

Recommended order:

- \(SS\): analytic reduced-space pole convolution;
- \(SR/RS\): projected pole-aware convolution;
- \(RR\): existing FFT/banded kernel on the coarse regular grid.

## Step 7 — reconstruct retarded self-energy

Use
\[
\Sigma_{\mathrm{analytic}}^R
+
\text{Hilbert transform of regular remainder}.
\tag{192}
\]

## Step 8 — conservation tests

Compute
\[
J_s,\qquad
J_L+J_R,\qquad
\text{telescoped current error}.
\tag{193}
\]

## Step 9 — mix / Newton update

Apply the existing damped, Anderson, or Newton–Krylov iteration to the hybrid state.

Potential reduced state:
\[
x
=
\left\{
z_\alpha,
R_\alpha,
S_{\alpha\beta}^{<,>},
\Sigma_{\mathrm{reg}}^{<,>}
\right\}.
\tag{194}
\]

## Step 10 — sector adaptation

Only at an adaptation boundary:

- inspect frequency/spatial/error indicators;
- promote/demote modes with hysteresis;
- migrate state;
- restart the next epoch.

---

# 45. Migration when a mode enters the pole sector

Suppose a pole is detected in the regular numerical representation.

1. compute \(z_\alpha,R_\alpha\);
2. construct its analytic pole function;
3. subtract it from the stored retarded regular Green function:
\[
G_R^R
\leftarrow
G_R^R
-
\frac{R_\alpha}{z-z_\alpha};
\tag{195}
\]
4. reconstruct/subtract the corresponding pole-containing Keldysh part;
5. verify the remainder is smoother;
6. recompute only the affected projected Gram matrices.

The migration must be done so that
\[
G_{\mathrm{before}}
=
G_{\mathrm{after}}
\tag{196}
\]
to within the representation tolerance.

---

# 46. Migration when a pole leaves the pole sector

If a linewidth becomes broad enough that the coarse grid resolves it:

1. evaluate the analytic pole contribution on the regular grid;
2. add it into the regular representation;
3. remove the pole object;
4. verify the reconstructed \(G\) is unchanged;
5. only then continue SCBA.

This makes promotion/demotion a change of representation, not a physical perturbation.

---

# 47. Practical staged implementation plan

The full method should not be implemented in one step.

## Phase 0 — baseline test harness

Before adding the hybrid method, store reference outputs from the existing solver for:

- one harmonic chain;
- one small anharmonic chain;
- a device with a quasi-bound sharp resonance;
- a device with a propagating continuum;
- a device near a band edge.

Reference:

\[
G^R,\quad G^{<,>},\quad
\Sigma^{R,<,>},\quad
J_L,J_R,J_s.
\]

---

## Phase 1 — retarded pole extraction only

Implement:

- complex-frequency \(M(z)\);
- nonlinear pole solve;
- left/right residue;
- direct reconstruction of selected \(G^R\) matrix elements.

No SCBA acceleration yet.

Acceptance test:
\[
G^R_{\mathrm{direct}}
\simeq
G^R_{\mathrm{pole}}+G^R_{\mathrm{bg}}
\tag{197}
\]
on dense audit points.

---

## Phase 2 — robust pole continuation

Implement:

- predictor Eq. (43);
- bordered Newton Eq. (46);
- clustering;
- principal-angle matching;
- contour fallback.

Stress test pole crossings and near-degenerate resonances.

---

## Phase 3 — \(G_{PP}^{<,>}\) with full coherence matrix

Implement Eq. (83).

First use direct interpolation of the projected source.

Validate:

- Keldysh identity;
- bosonic frequency symmetry;
- equilibrium detailed balance;
- semidefinite spectral/occupation structure.

---

## Phase 4 — analytic \(SS\) bubble

Implement Eqs. (113)–(117).

Compare against a deliberately over-resolved FFT convolution.

This phase should already demonstrate whether the frequency-grid bottleneck is substantially reduced.

---

## Phase 5 — mixed \(SR/RS\) terms

First implementation:

- keep pole denominators analytic;
- evaluate projected regular coefficient functions on a coarse grid;
- use adaptive pole-aware quadrature.

Only after correctness is established, replace the adaptive quadrature with rational fits/residue sums.

---

## Phase 6 — regular-grid coarsening

After the pole-containing terms are removed, measure the smoothness of
\[
G_R^{<,>}.
\]

Increase
\[
h_{\mathrm{reg}}
\]
until the hybrid self-energy/current error reaches the target tolerance.

This measures the actual acceleration available from pole subtraction.

---

## Phase 7 — spatial modal factorization

Add complex-band tracking and
\[
G_S(i,j)=U_iCV_j^\dagger.
\]

Initially use it only to reconstruct distant blocks; do not alter the SCBA kernel.

Validate against exact distant \(G_{ij}\).

---

## Phase 8 — modal/banded SCBA contraction

Use factorized \(G_S\) inside the FC3 ring.

Combine with the report's CP/INDSCAL/Waring factorization only after the dense-vertex version is correct.

---

## Phase 9 — dynamic sector adaptation

Only now add automatic promotion/demotion.

Before this stage, sector membership should be manually specified for controlled tests.

---

# 48. Complexity expectations

Let

- \(N_\omega\): old fine-grid size;
- \(N_\omega^{R}\): regular coarse-grid size;
- \(N_p\): number of pole modes;
- \(r_x\): number of long-range spatial modes;
- \(b_R\): regular spatial band;
- \(R_\Phi\): FC3 decomposition rank.

The current regular-grid cost remains approximately the report's existing kernel cost but with
\[
N_\omega
\rightarrow
N_\omega^R
\tag{198}
\]
and potentially
\[
b_G
\rightarrow
b_R.
\tag{199}
\]

The pole-pole part scales with mode pairs rather than the fine frequency grid:
\[
\text{work}_{SS}
\sim
O(N_p^2)
\tag{200}
\]
times the cost of the reduced vertex/output contractions.

With factorized FC3, the expensive dependence on the full block size is replaced largely by \(R_\Phi\)-dimensional Gram operations.

The long-range spatial memory changes schematically from
\[
O(N_B^2N_{BS}^2)
\tag{201}
\]
for explicit dense distant blocks to approximately
\[
O(N_BN_{BS}r_x)
\tag{202}
\]
plus small modal matrices, whenever the low-rank propagation form is valid.

The important point is not only the asymptotic count. Pole subtraction changes the fundamental required time/frequency information scale from
\[
1/\Gamma_{\min}
\]
to
\[
1/\Gamma_{\mathrm{reg}}.
\tag{203}
\]

That is a real reduction in information content, unlike a nonuniform FFT that internally still resolves the original narrow linewidth.

---

# 49. What should *not* be done

## Do not project propagating modes out of SCBA

That removes their anharmonic scattering channels.

## Do not drop \(SR\) and \(RS\)

Those are physical pole-background three-phonon processes.

## Do not treat raw \(G^{<,>}\) as causal analytic functions

The simple residue theorem applies directly to retarded/advanced functions. For lesser/greater functions, first derive their rational structure from Keldysh.

## Do not assign one scalar occupation per pole by default

Keep the cluster matrix
\[
S_{\alpha\beta}^{<,>}
\]
until coherence is demonstrated to be negligible.

## Do not freeze first-iteration plane waves under a nonuniform SCBA self-energy

Use a reduced modal basis plus coupling/Schur complement.

## Do not force band-edge continua into isolated poles

Treat them as continuum/modal or multi-pole rational sectors.

## Do not change sector membership every iteration

Use hysteretic outer adaptation epochs.

## Do not separately approximate the two bubble legs

A conserving SCBA requires the same reconstructed \(G\) on both internal lines.

---

# 50. Reassessment of the three research difficulties

## 50.1 Robust pole tracking

### Earlier concern

> Poles can move, cross, broaden, merge, or disappear during SCBA.

### Revised assessment

This is a well-posed nonlinear-eigenvalue continuation problem and has a clear robust solution:

\[
\boxed{
\text{predictor}
+
\text{bordered Newton}
+
\text{subspace tracking}
+
\text{contour fallback}.
}
\]

The difficult cases are not ordinary moving poles but

- branch-edge resonances;
- defective/exceptional clusters;
- abrupt changes caused by a large SCBA step.

These should be detected and moved to a cluster/continuum representation.

**Conclusion:** pole tracking is an engineering challenge, not a fundamental obstacle.

---

## 50.2 Nonequilibrium \(G^{<,>}\) pole weights and coherences

### Earlier concern

> There is no equilibrium Bose factor that can simply be attached to each pole under bias.

### Revised assessment

Correct. The proper object is
\[
\boxed{
S^{\lessgtr}
=
V^\dagger\Sigma_{\mathrm{tot}}^{\lessgtr}V,
}
\]
and
\[
\boxed{
G_{PP}^{\lessgtr}
=
U D^R S^{\lessgtr}D^A U^\dagger.
}
\]

This automatically retains

- nonequilibrium injection;
- scattering-generated populations;
- modal coherences;
- non-Hermitian left/right structure.

The remaining challenge is how accurately \(S^{<,>}\) must be represented across each pole cluster.

A local rational or polynomial representation solves that without restoring a global fine grid.

**Conclusion:** this is the most important physics/numerics issue, but it has a clean formulation. The wrong solution is scalar occupations; the right solution is a small Keldysh matrix per pole cluster.

---

## 50.3 Dynamic modal-versus-regular selection

### Earlier concern

> A fixed manual choice may fail as linewidths and decay lengths change during self-consistency.

### Revised assessment

The classification should be based on explicit representation-error criteria:

\[
\gamma_\alpha
\quad\text{vs.}\quad
h_{\mathrm{reg}},
\]
\[
|\lambda_\alpha|^{b_R}
\quad\text{vs.}\quad
\epsilon_x,
\]
\[
\eta_\alpha
\quad\text{for isolation},
\]
\[
\kappa_\alpha
\quad\text{for conditioning},
\]
and
\[
I_\alpha
\quad\text{for bubble relevance}.
\]

Use a three-way classification:

\[
\boxed{
\text{isolated pole}
\quad/\quad
\text{long-range continuum mode}
\quad/\quad
\text{regular remainder}.
}
\]

Use hysteresis and adaptation epochs so the fixed-point map does not jump every iteration.

**Conclusion:** the selection can be made controlled and reproducible rather than heuristic.

---

# 51. A fourth difficulty that should be stated explicitly

The main additional caveat is **analytic continuation of the open-boundary contact problem**.

For a finite closed device the pole decomposition is straightforward.

For an open device:

- the contact Green function has branch cuts;
- in-band resonances may live on a continued sheet;
- band edges are branch points;
- propagating continua cannot be replaced exactly by finitely many isolated poles.

Therefore the strongest production formulation is not a pure "sum of poles."

It is:

\[
\boxed{
\text{isolated poles}
+
\text{modal continuum}
+
\text{short-range regular background}.
}
\]

This is the formulation that should guide the implementation.

---

# 52. Recommended first production target

A realistic first target is **not** the fully adaptive three-sector algorithm.

Implement:

1. a manually selected set of isolated sharp poles;
2. exact retarded pole subtraction;
3. full cluster \(G_{PP}^{<,>}\);
4. analytic \(SS\) bubble;
5. pole-aware \(SR+RS\);
6. existing FFT \(RR\);
7. no automatic spatial mode reduction yet.

If that reproduces the dense SCBA while allowing
\[
N_\omega^R
\ll
N_\omega,
\]
the core idea is validated.

Then add the spatial modal representation.

---

# 53. Minimal numerical experiments required to establish the method

## Experiment A — single isolated resonance

Artificially construct one sharp mode with known
\[
\Omega,\gamma.
\]

Verify:

- extracted pole;
- residue;
- Lorentzian Keldysh weight;
- analytic self-convolution;
- exact reconstruction on a coarse grid.

## Experiment B — two nearby modes

Vary their separation through
\[
\Delta\Omega\sim\gamma_1+\gamma_2.
\]

Show where scalar occupations fail and the cluster matrix succeeds.

## Experiment C — pole crossing during SCBA

Drive two mode frequencies through one another between artificial iterations.

Compare:

- frequency sorting;
- vector overlap sorting;
- cluster principal-angle tracking.

## Experiment D — propagating + evanescent spatial mixture

Construct
\[
G_n
=
A e^{iqn}
+
B e^{-\kappa n}.
\]

Show:

- a hard band fails;
- modal + banded decomposition is exact;
- geometric summation reproduces long-range terms.

## Experiment E — contact band edge

Place a resonance close to a lead band edge.

Show that a single pole fit loses accuracy and the continuum/rational representation is required.

## Experiment F — conserving SCBA

For a small device compare dense and hybrid calculations of

\[
J_s,\qquad
J_L+J_R,\qquad
\Sigma^R-\Sigma^A-(\Sigma^>-\Sigma^<).
\]

The hybrid method should converge to the same values within the chosen representation tolerance.

---

# 54. Compact algorithm pseudocode

```text
initialize ballistic/contact problem

build initial regular frequency grid
find candidate sharp resonances
build initial pole clusters
build initial spatial slow-mode set

for adaptation_epoch:

    freeze sector membership

    for SCBA iteration n:

        # retarded operator
        construct M_n(z)

        # track isolated poles
        for each pole cluster:
            predict z using nonlinear sensitivity
            correct using bordered Newton
            if failure / pole count mismatch:
                contour-rescan cluster
            update left/right vectors and residues

        # update long-range modal subspace
        track propagating/slow complex-band modes
        update reduced modal basis / Schur complement

        # construct hybrid retarded G
        G_pole  = analytic pole representation
        G_cont  = modal continuum representation
        G_reg   = local/banded numerical remainder

        # Keldysh
        project Sigma_tot^{<,>} into pole clusters
        build full cluster G_PP^{<,>}
        build pole-background mixed pieces
        build regular G_BB^{<,>}

        # bubble
        Sigma_SS = analytic/reduced pole convolution
        Sigma_SR = pole-aware projected convolution
        Sigma_RS = matching transpose/partner evaluation
        Sigma_RR = existing FFT/banded kernel

        Sigma_new^{<,>} = SS + SR + RS + RR

        # retarded reconstruction
        Sigma_new^R =
            analytic pole retarded part
            + KK/Hilbert regular remainder

        # mix / accelerate
        update SCBA state

        # diagnostics
        check Keldysh identity
        check bosonic symmetry
        check causality
        check Js
        check JL + JR
        check current continuity
        check pole reconstruction errors

        if converged:
            break

    evaluate classification indicators

    if no promotion/demotion needed:
        finish

    migrate promoted/demoted modes
    restart next adaptation epoch
```

---

# 55. Final assessment

The decomposition is analytically sound **with the following precise meaning**:

\[
\boxed{
G^R
=
G_{\mathrm{pole}}^R
+
G_{\mathrm{cont}}^R
+
G_{\mathrm{reg}}^R
}
\]
is an exact or controlled representation when

- isolated pole residues are computed with the nonlinear left/right normalization;
- contact branch cuts are kept in a continuum/background sector unless explicitly analytically continued;
- the Keldysh functions are derived from the retarded decomposition rather than assigned ad hoc occupations;
- pole coherences are retained as small matrices;
- all \(SS,SR,RS,RR\) bubble contributions are included;
- the spatial long-range subspace remains coupled to the regular subspace through a Schur complement;
- the same reconstructed \(G\) is used on both lines of the conserving SCBA diagram.

The key numerical idea can therefore be summarized as

\[
\boxed{
\begin{array}{c}
\text{narrow frequency structure}
\\
\downarrow
\\
\text{analytic poles / small pole clusters}
\end{array}
}
\qquad
+
\qquad
\boxed{
\begin{array}{c}
\text{long-range spatial structure}
\\
\downarrow
\\
\text{factorized modal propagation}
\end{array}
}
\qquad
+
\qquad
\boxed{
\begin{array}{c}
\text{smooth, decaying remainder}
\\
\downarrow
\\
\text{coarse-grid banded SCBA}
\end{array}
}
\]

The residue theorem is not the limiting issue. The real work is to make the decomposition **stable under self-consistency and faithful to nonequilibrium Keldysh structure**. The procedures above provide a concrete route to doing that.

---

# 56. References and relation to the attached report

## Attached report

**Anharmonic Phonon-Phonon Interactions — Project Notes, 10 Aug. 2026.**

Most directly relevant sections:

- Sec. 1.2.2: Dyson and Keldysh equations.
- Sec. 1.2.4: contact surface Green function from propagating/evanescent modes.
- Sec. 1.3.4: cubic lesser/greater bubble.
- Sec. 1.3.5: retarded reconstruction and principal-value term.
- Sec. 1.4.3: analytic SCBA Jacobian / Newton–Krylov.
- Sec. 1.4.4: sharp resonances and grid-resolution criterion.
- Sec. 1.4.5: conserving SCBA and energy-balance diagnostics.
- Sec. 1.5.1: FFT convolution and bosonic fold.
- Sec. 1.5.2: nonuniform grids and pole representations.
- Sec. 1.5.5–1.5.6: non-banded propagating Green functions and causal spatial truncation.
- Sec. 1.6: factorized FC3 self-energy kernels.

## External mathematical / NEGF references used in developing the proposal

1. W.-J. Beyn, **An integral method for solving nonlinear eigenvalue problems**, *Linear Algebra and its Applications* **436**, 3839–3863 (2012). DOI: `10.1016/j.laa.2011.03.030`.

2. M. C. Brennan, M. Embree, and S. Gugercin, **Contour Integral Methods for Nonlinear Eigenvalue Problems: A Systems Theoretic Approach**, *SIAM Review* **65**, 439–470 (2023). DOI: `10.1137/20M1389303`.

3. A. L. Andrew, K.-W. E. Chu, and P. Lancaster, **Derivatives of Eigenvalues and Eigenvectors of Matrix Functions**, *SIAM Journal on Matrix Analysis and Applications* **14**, 903–926 (1993). DOI: `10.1137/0614061`.

4. Z. Drmač, **On Principal Angles between Subspaces of Euclidean Space**, *SIAM Journal on Matrix Analysis and Applications* **22**, 173–194 (2000). DOI: `10.1137/S0895479897320824`.

5. G. Baym and L. P. Kadanoff, **Conservation Laws and Correlation Functions**, *Physical Review* **124**, 287 (1961). DOI: `10.1103/PhysRev.124.287`.

6. G. Baym, **Self-Consistent Approximations in Many-Body Systems**, *Physical Review* **127**, 1391 (1962). DOI: `10.1103/PhysRev.127.1391`.

7. Y. Xu, J.-S. Wang, W. Duan, B.-L. Gu, and B. Li, **Nonequilibrium Green's function method for phonon-phonon interactions and ballistic-diffusive thermal transport**, *Physical Review B* **78**, 224303 (2008). DOI: `10.1103/PhysRevB.78.224303`.

8. H. H. B. Sørensen, P. C. Hansen, D. E. Petersen, S. Skelboe, and K. Stokbro, **Efficient wave-function matching approach for quantum transport calculations**, *Physical Review B* **79**, 205322 (2009).

9. Y. Dewulf, D. Van Neck, and M. Waroquier, **Discrete approach to self-consistent GW calculations in an electron gas**, *Physical Review B* **71**, 245122 (2005). DOI: `10.1103/PhysRevB.71.245122`. This is a useful precedent for replacing a complicated self-consistent Green-function energy dependence by a finite, carefully managed pole representation.

---

## Short implementation recommendation

If only one part is implemented first, implement:

\[
\boxed{
\text{retarded pole subtraction}
\rightarrow
\text{full pole-cluster Keldysh matrix}
\rightarrow
\text{analytic }SS
\rightarrow
\text{pole-aware }SR+RS
\rightarrow
\text{coarse-grid }RR.
}
\]

That path directly tests whether the method removes the linewidth-driven frequency-grid cost **without yet changing the spatial solver**. If successful, the modal/banded spatial decomposition is the second major optimization.
