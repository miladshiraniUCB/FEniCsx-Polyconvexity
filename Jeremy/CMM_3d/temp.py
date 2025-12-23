import ufl
from dolfinx import fem

# ---------- helpers ----------
def unit(v, eps=1e-12):
    return v / ufl.sqrt(ufl.dot(v, v) + eps)

def fung_fiber_energy(C_xiN, a_xiN, c1, c2):
    # Eq. (52): Ŵ^ξ = c1/(4 c2) * (exp(c2*(C:(a⊗a)-1)^2) - 1)
    I4 = ufl.dot(a_xiN, ufl.dot(C_xiN, a_xiN))  # C:(a⊗a)
    E  = I4 - 1.0
    return (c1 / (4.0 * c2)) * (ufl.exp(c2 * E**2) - 1.0)

def prestretch_isochoric(a, g):
    """
    Build a symmetric, volume-preserving (det=1) prestretch tensor with
    stretch g along direction a and 1/sqrt(g) in transverse directions.
    This matches the paper’s assumption that deposition tensors are symmetric
    and volume-preserving. :contentReference[oaicite:3]{index=3}
    """
    I  = ufl.Identity(3)
    aa = ufl.outer(a, a)
    return g * aa + (1.0 / ufl.sqrt(g)) * (I - aa)

def C_with_prestretch(C, G):
    # general: Cα = (F G)^T (F G) = G^T C G
    return G.T * C * G

# ---------- kinematics ----------
u  = fem.Function(V)               # displacement
v  = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

x = ufl.SpatialCoordinate(mesh)
I = ufl.Identity(3)
F = I + ufl.grad(u)
C = ufl.variable(F.T * F)
J = ufl.det(F)

# ---------- local cylindrical basis (tube axis along z) ----------
ez = ufl.as_vector((0.0, 0.0, 1.0))
er = unit(ufl.as_vector((x[0], x[1], 0.0)))
etheta = ufl.cross(ez, er)

# ---------- material/mixture parameters (examples as Constants) ----------
phi_e0 = fem.Constant(mesh, 0.34)
phi_m0 = fem.Constant(mesh, 0.33)
phi_c0 = fem.Constant(mesh, 0.33)

# collagen family fractions beta_θ, beta_z, beta_d (sum=1 over collagen families)
beta_th = fem.Constant(mesh, 0.056)
beta_z  = fem.Constant(mesh, 0.067)
beta_d  = fem.Constant(mesh, 0.877)

# Eq. (51) elastin matrix parameter
c_e = fem.Constant(mesh, 89.0)  # example

# Eq. (52) fiber parameters for muscle/collagen
c1_m = fem.Constant(mesh, 10.0); c2_m = fem.Constant(mesh, 50.0)
c1_c = fem.Constant(mesh, 10.0); c2_c = fem.Constant(mesh, 50.0)

# deposition stretches (examples; in the paper they are treated as constants at s=0) :contentReference[oaicite:4]{index=4}
Gm = fem.Constant(mesh, 1.20)
Gc = fem.Constant(mesh, 1.25)

# elastin prestretches (example; paper lists circumferential/axial and uses isochoric constraint) :contentReference[oaicite:5]{index=5}
Gth_e = fem.Constant(mesh, 1.90)
Gz_e  = fem.Constant(mesh, 1.62)
Gr_e  = 1.0 / (Gth_e * Gz_e)      # det=1

# ---------- build prestretch tensors ----------
# elastin matrix prestretch tensor in cylindrical basis
G_e = (Gr_e  * ufl.outer(er, er)
     + Gth_e * ufl.outer(etheta, etheta)
     + Gz_e  * ufl.outer(ez, ez))

# muscle is circumferential in the four-fiber family model :contentReference[oaicite:6]{index=6}
a_m = etheta
G_mN = prestretch_isochoric(a_m, Gm)

# collagen families: circumferential, axial, and two symmetric diagonals (±) with angle α0 to axial :contentReference[oaicite:7]{index=7}
alpha0 = fem.Constant(mesh, 29.91 * (ufl.pi/180.0))  # example from Table 1 :contentReference[oaicite:8]{index=8}

a_c_th = etheta
a_c_z  = ez
a_c_d1 = unit( ufl.sin(alpha0)*etheta + ufl.cos(alpha0)*ez )
a_c_d2 = unit(-ufl.sin(alpha0)*etheta + ufl.cos(alpha0)*ez )

# (optional) one Gc for all collagen families, as in the paper’s baseline list
G_cN_th = prestretch_isochoric(a_c_th, Gc)
G_cN_z  = prestretch_isochoric(a_c_z,  Gc)
G_cN_d1 = prestretch_isochoric(a_c_d1, Gc)
G_cN_d2 = prestretch_isochoric(a_c_d2, Gc)

# ---------- constituent C-tensors ----------
C_e = C_with_prestretch(C, G_e)
C_m = C_with_prestretch(C, G_mN)
C_c_th = C_with_prestretch(C, G_cN_th)
C_c_z  = C_with_prestretch(C, G_cN_z)
C_c_d1 = C_with_prestretch(C, G_cN_d1)
C_c_d2 = C_with_prestretch(C, G_cN_d2)

# ---------- energies ----------
# Eq. (51): Ŵ^e = (c_e/2)*(C_e:I - 3) :contentReference[oaicite:9]{index=9}
W_e = 0.5 * c_e * (ufl.tr(C_e) - 3.0)

# Eq. (52): fiber energies :contentReference[oaicite:10]{index=10}
W_m = fung_fiber_energy(C_m, a_m, c1_m, c2_m)

W_c = (
    beta_th * fung_fiber_energy(C_c_th, a_c_th, c1_c, c2_c)
  + beta_z  * fung_fiber_energy(C_c_z,  a_c_z,  c1_c, c2_c)
  + 0.5*beta_d * (fung_fiber_energy(C_c_d1, a_c_d1, c1_c, c2_c)
                + fung_fiber_energy(C_c_d2, a_c_d2, c1_c, c2_c))
)

# near-incompressibility penalty U(ln J)
kappa = fem.Constant(mesh, 1e3)         # pick >> shear scale
Uvol  = 0.5 * kappa * (ufl.ln(J))**2

W = phi_e0*W_e + phi_m0*W_m + phi_c0*W_c + Uvol

# external work example (pressure p on lumen):  Π_ext = ∫ p * (u·n) ds
# (define ds and n for your tagged lumen facet)
# Pi = ufl.integral(W*ufl.dx) - p*ufl.dot(u, n)*ds(lumen_id)

Pi = W * ufl.dx  # placeholder if you haven’t added loads yet

R = ufl.derivative(Pi, u, v)
Jform = ufl.derivative(R, u, du)
