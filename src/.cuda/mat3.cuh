
typedef struct {
	REAL c[6];

	inline __host__ __device__ REAL3 c0() const { return make_REAL3(c[0], c[1], c[2]); }
	inline __host__ __device__ REAL3 c1() const { return make_REAL3(c[3], c[4], c[5]); }
} REAL3x2;

typedef struct {
	REAL c[6];

	inline __host__ __device__ REAL2 c0() const { return make_REAL2(c[0], c[1]); }
	inline __host__ __device__ REAL2 c1() const { return make_REAL2(c[2], c[3]); }
	inline __host__ __device__ REAL2 c2() const { return make_REAL2(c[4], c[5]); }
} REAL2x3;

typedef struct {
	REAL c[4];

	inline __host__ __device__ REAL2 c0() const { return make_REAL2(c[0], c[1]); }
	inline __host__ __device__ REAL2 c1() const { return make_REAL2(c[2], c[3]); }
} REAL2x2;

inline __host__ __device__ REAL3x2 make_REAL3x2(REAL3 c0, REAL3 c1)
{
	REAL3x2 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c0.z;
	t.c[3] = c1.x;
	t.c[4] = c1.y;
	t.c[5] = c1.z;
	return t;
}

inline __host__ __device__ REAL2x3 make_REAL2x3(REAL2 c0, REAL2 c1, REAL2 c2)
{
	REAL2x3 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c1.x;
	t.c[3] = c1.y;
	t.c[4] = c2.x;
	t.c[5] = c2.y;
	return t;
}

inline __host__ __device__ REAL2x2 make_REAL2x2(REAL2 c0, REAL2 c1)
{
	REAL2x2 t;

	t.c[0] = c0.x;
	t.c[1] = c0.y;
	t.c[2] = c1.x;
	t.c[3] = c1.y;
	return t;
}

inline __host__ __device__ REAL det(REAL2x2 t)
{
	return t.c[0] * t.c[3] - t.c[1] * t.c[2];
}

inline __host__ __device__ REAL2x2 inverse(const REAL2x2 &t)
{
	REAL detInv = 1.0f / det(t);

	return make_REAL2x2(
		make_REAL2( t.c[3],  -t.c[1])*detInv,
		make_REAL2( -t.c[2],  t.c[0])*detInv);
}

inline __host__ __device__ REAL2x3 operator* (const REAL2x2 &a, const REAL2x3 &b)
{
	return make_REAL2x3(
		make_REAL2(
		a.c[0]*b.c[0]+a.c[2]*b.c[1],
		a.c[1]*b.c[0]+a.c[3]*b.c[1]),
		make_REAL2(
		a.c[0]*b.c[2]+a.c[2]*b.c[3],
		a.c[1]*b.c[2]+a.c[3]*b.c[3]),
		make_REAL2(
		a.c[0]*b.c[4]+a.c[2]*b.c[5],
		a.c[1]*b.c[4]+a.c[3]*b.c[5])
	);
}

inline __host__ __device__ REAL3x2 operator* (const REAL3x2 &a, const REAL2x2 &b)
{
	return make_REAL3x2(
		make_REAL3(
		a.c[0]*b.c[0]+a.c[3]*b.c[1],
		a.c[1] * b.c[0] + a.c[4] * b.c[1],
		a.c[2] * b.c[0] + a.c[5] * b.c[1]),
		make_REAL3(
		a.c[0]*b.c[2]+a.c[3]*b.c[3],
		a.c[1] * b.c[2] + a.c[4] * b.c[3],
		a.c[2] * b.c[2] + a.c[5] * b.c[3])
	);
}

inline __host__ __device__ REAL3 operator* (const REAL3x2 &a, const REAL2 &b)
{
	return make_REAL3(
		a.c[0] * b.x + a.c[3] * b.y,
		a.c[1] * b.x + a.c[4] * b.y,
		a.c[2] * b.x + a.c[5] * b.y);
}

/////////////////////////////////////////////////////////////////////////////////////////////
typedef struct {
	REAL c[9];

	inline __host__ __device__ REAL3 c0() const { return make_REAL3(c[0], c[1], c[2]); }
	inline __host__ __device__ REAL3 c1() const { return make_REAL3(c[3], c[4], c[5]); }
	inline __host__ __device__ REAL3 c2() const { return make_REAL3(c[6], c[7], c[8]); }

	inline __host__ __device__ void put(REAL *ptr) {
		for (int i = 0; i < 9; i++)
			c[i] = ptr[i];
	}

	inline __host__ __device__ void get(REAL *ptr) const {
		for (int i = 0; i < 9; i++)
			ptr[i] = c[i];
	}
} REAL3x3;

typedef struct __align__(16) {
    REAL3x3 u;
    REAL2 s;
    REAL2x2 vt;
} svd3x2;

inline __host__ __device__ REAL3 getCol(REAL3x3 a, int i)
{
	if (i == 0)
		return a.c0();
	else if (i == 1)
		return a.c1();
	else
		return a.c2();
}

inline __host__ __device__ REAL3x3 make_REAL3x3 (REAL3 c0, REAL3 c1, REAL3 c2) {
	REAL3x3 t;
	t.c[0] = c0.x, t.c[1] = c0.y, t.c[2] = c0.z;
	t.c[3] = c1.x, t.c[4] = c1.y, t.c[5] = c1.z;
	t.c[6] = c2.x, t.c[7] = c2.y, t.c[8] = c2.z;
	return t;
}

inline __host__ __device__ void setIJ(REAL3x2 &m, int i, int j, REAL v)
{
	m.c[j * 3 + i] = v;
}

inline __host__ __device__ REAL getIJ(const REAL3x3 &m, int i, int j)
{
	return m.c[j * 3 + i];
}

inline __host__ __device__ REAL &getIJ(REAL3x3 &m, int i, int j)
{
	return m.c[j * 3 + i];
}

inline __host__ __device__ REAL getIJ(const REAL2x2 &m, int i, int j)
{
	return m.c[j * 2 + i];
}

inline __host__ __device__ REAL getI(const REAL2x2 &m, int i)
{
	return m.c[i];
}

inline __host__ __device__ REAL &getI(REAL2x2 &m, int i)
{
	return m.c[i];
}

inline __host__ __device__ REAL getI(const REAL3x3 &m, int i)
{
	int id = i/3;
	return getI(getCol(m, id), i-id*3);
}

inline __host__ void print_REAL3x3 (REAL3x3 * m) {
	printf("%f ", m->c[0]); printf("%f ", m->c[3]); printf("%f \n", m->c[6]); 
	printf("%f ", m->c[1]); printf("%f ", m->c[4]); printf("%f \n", m->c[7]); 
	printf("%f ", m->c[2]); printf("%f ", m->c[5]); printf("%f \n\n", m->c[8]); 
}

inline __host__ __device__ REAL3x3 operator+ (const REAL3x3 &m1, const REAL3x3 &m2)
{
	return make_REAL3x3(
		m1.c0()+ m2.c0(),
		m1.c1()+ m2.c1(),
		m1.c2()+ m2.c2());
}

inline __host__ __device__ void operator+= (REAL3x3 &m1, const REAL3x3 &m2)
{
	for (int i = 0; i < 9; i++)
		m1.c[i] += m2.c[i];
}


inline __host__ __device__ REAL3x3 operator- (const REAL3x3 &m1, const REAL3x3 &m2) {
	return make_REAL3x3(
		m1.c0() - m2.c0(),
		m1.c1() - m2.c1(),
		m1.c2() - m2.c2());
}

inline __host__ __device__ REAL2x2 operator- (const REAL2x2 &m1, const REAL2x2 &m2) {
	return make_REAL2x2(
		m1.c0() - m2.c0(),
		m1.c1() - m2.c1());
}

inline __host__ __device__ REAL3x3 operator- (const REAL3x3 &a) {
	return make_REAL3x3(-a.c0(), -a.c1(), -a.c2());
}


inline __host__ __device__ REAL3x3 operator* (REAL a, const REAL3x3 &m) {
	return make_REAL3x3(a * m.c0(), a * m.c1(),  a * m.c2());
}

inline __host__ __device__ REAL3x3 operator* (const REAL3x3 &m, REAL a) {
	return make_REAL3x3(a * m.c0(), a * m.c1(), a * m.c2());
}

inline __host__ __device__ void operator*= (REAL3x3 &m, REAL a) {
	for (int i = 0; i < 9; i++)
		m.c[i] *= a;
}

inline __host__ __device__ void operator*= (REAL2x2 &m, REAL a) {
	for (int i = 0; i < 4; i++)
		m.c[i] *= a;
}

inline __host__ __device__ REAL2x2 operator* (REAL a, const REAL2x2 &m) {
	return make_REAL2x2(a * m.c0(), a * m.c1());
}

inline __host__ __device__ REAL2x2 operator* (const REAL2x2 &m, REAL a) {
	return make_REAL2x2(a * m.c0(), a * m.c1());
}

inline __host__ __device__ REAL3x3 getTrans (const REAL3x3 &m) {
	REAL3x3 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[3];
	t.c[2] = m.c[6];
	t.c[3] = m.c[1];
	t.c[4] = m.c[4];
	t.c[5] = m.c[7];
	t.c[6] = m.c[2];
	t.c[7] = m.c[5];
	t.c[8] = m.c[8];
	return t;
}

inline __host__ __device__ void getTrans(REAL t[], const REAL m[]) {
	t[0] = m[0];
	t[1] = m[3];
	t[2] = m[6];
	t[3] = m[1];
	t[4] = m[4];
	t[5] = m[7];
	t[6] = m[2];
	t[7] = m[5];
	t[8] = m[8];
}

inline __host__ __device__ REAL2x3 getTrans (const REAL3x2 &m) {
	REAL2x3 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[3];
	t.c[2] = m.c[1];
	t.c[3] = m.c[4];
	t.c[4] = m.c[2];
	t.c[5] = m.c[5];
	return t;
}

inline __host__ __device__ REAL3x2 getTrans (const REAL2x3 &m) {
	REAL3x2 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[2];
	t.c[2] = m.c[4];
	t.c[3] = m.c[1];
	t.c[4] = m.c[3];
	t.c[5] = m.c[5];
	return t;
}

inline __host__ __device__ REAL2x2 getTrans (const REAL2x2 &m) {
	REAL2x2 t;
	t.c[0] = m.c[0];
	t.c[1] = m.c[2];
	t.c[2] = m.c[1];
	t.c[3] = m.c[3];
	return t;
}

inline __host__ __device__ REAL3x3 operator* (const REAL3x3 &m1, const REAL3x3 &m2) {
	REAL3x3 m1T = getTrans(m1);

	return make_REAL3x3(
		make_REAL3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_REAL3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())),
		make_REAL3(dot(m1T.c0(), m2.c2()), dot(m1T.c1(), m2.c2()), dot(m1T.c2(), m2.c2())));
}

inline __host__ __device__ REAL3x2 operator* (const REAL3x3 &m1, const REAL3x2 &m2)
{
	REAL3x3 m1T = getTrans(m1);
	return make_REAL3x2(
		make_REAL3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_REAL3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())));
}

inline __host__ __device__ REAL2x2 operator* (const REAL2x3 &m1, const REAL3x2 &m2) {
	REAL3x2 m1T = getTrans(m1);

	return make_REAL2x2(
		make_REAL2(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0())),
		make_REAL2(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1())));
}

inline __host__ __device__ REAL3x3 operator* (const REAL3x2 &m1, const REAL2x3 &m2) {
	REAL2x3 m1T = getTrans(m1);

	return make_REAL3x3(
		make_REAL3(dot(m1T.c0(), m2.c0()), dot(m1T.c1(), m2.c0()), dot(m1T.c2(), m2.c0())),
		make_REAL3(dot(m1T.c0(), m2.c1()), dot(m1T.c1(), m2.c1()), dot(m1T.c2(), m2.c1())),
		make_REAL3(dot(m1T.c0(), m2.c2()), dot(m1T.c1(), m2.c2()), dot(m1T.c2(), m2.c2())));
}

inline __host__ __device__ REAL3x3 outer(const REAL3 &a, const REAL3 &b){
	return make_REAL3x3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ REAL3x3 identity3x3(){
	return make_REAL3x3(
					   make_REAL3(1.0, 0.0, 0.0),
					   make_REAL3(0.0, 1.0, 0.0),
					   make_REAL3(0.0, 0.0, 1.0));
}

inline __host__ __device__ REAL2x2 identity2x2(){
	return make_REAL2x2(
					   make_REAL2(1.0, 0.0),
					   make_REAL2(0.0, 1.0));
}

inline __host__ __device__ REAL3x3 make_REAL3x3(REAL m){
	return make_REAL3x3(
					   make_REAL3(m, 0.0, 0.0),
					   make_REAL3(0.0, m, 0.0),
					   make_REAL3(0.0, 0.0, m)
					   );
}

inline __host__ __device__ REAL2x2 make_REAL2x2(REAL m){
	return make_REAL2x2(
					   make_REAL2(m, 0.0),
					   make_REAL2(0.0, m)
					   );
}

inline __host__ __device__ REAL3x3 zero3x3(){
	return make_REAL3x3(
					   make_REAL3(0.0, 0.0, 0.0),
					   make_REAL3(0.0, 0.0, 0.0),
					   make_REAL3(0.0, 0.0, 0.0)
					   );
}

inline __host__ __device__ REAL2x2 zero2x2()
{
	return make_REAL2x2(
		make_REAL2(0, 0),
		make_REAL2(0, 0));
}

inline __host__ __device__ REAL3x2 zero3x2()
{
	return make_REAL3x2(
		make_REAL3(0, 0, 0),
		make_REAL3(0, 0, 0));
}

inline __host__ __device__ REAL3 getRow(REAL2x3 m, int i)
{
	if (i == 0)
		return make_REAL3(m.c[0], m.c[2], m.c[4]);
	if (i == 1)
		return make_REAL3(m.c[1], m.c[3], m.c[5]);

	assert(0);
	return zero3f();
}

inline __host__ __device__ REAL3x3 getInverse (REAL3x3 &m) {
	REAL3x3 t;
	t.c[0] = m.c[4]*m.c[8] - m.c[5]*m.c[7];
	t.c[1] = m.c[7]*m.c[2] - m.c[8]*m.c[1];
	t.c[2] = m.c[1]*m.c[5] - m.c[2]*m.c[4];

	t.c[3] = m.c[5]*m.c[6] - m.c[3]*m.c[8];
	t.c[4] = m.c[8]*m.c[0] - m.c[6]*m.c[2];
	t.c[5] = m.c[2]*m.c[3] - m.c[0]*m.c[5];

	t.c[6] = m.c[3]*m.c[7] - m.c[4]*m.c[6];
	t.c[7] = m.c[6]*m.c[1] - m.c[7]*m.c[0];
	t.c[8] = m.c[0]*m.c[4] - m.c[1]*m.c[3];

	REAL det = m.c[0]*t.c[0] + m.c[1]*t.c[3] + m.c[2]*t.c[6];
	REAL detInv = 1.0f/det;

	for (int i = 0; i < 9; i++)
		t.c[i] *= detInv;

	return t;
}

inline __host__ __device__ REAL3x3 rotation(REAL3 &axis, REAL theta ) {
	REAL s = sin( theta );
	REAL c = cos( theta );
	REAL t = 1-c;
	REAL x = axis.x, y = axis.y, z = axis.z;
		
	REAL3x3 tt;
	tt.c[0] = t*x*x + c;
	tt.c[3] = t*x*y - s*z;
	tt.c[6] = t*x*z + s*y;
	tt.c[1]= t*x*y + s*z;
	tt.c[4] = t*y*y + c;
	tt.c[7] = t*y*z - s*x;
	tt.c[2] = t*x*z - s*y;
	tt.c[5] = t*y*z + s*x;
	tt.c[8] = t*z*z + c;

	return tt;
}
	
inline __host__ __device__ REAL3 operator* (const REAL3x3 &m, const REAL3 &rhs) {
			return make_REAL3(
			m.c[0]*rhs.x + m.c[3]*rhs.y + m.c[6]*rhs.z,
			m.c[1]*rhs.x + m.c[4]*rhs.y + m.c[7]*rhs.z,
			m.c[2]*rhs.x + m.c[5]*rhs.y + m.c[8]*rhs.z);
}

inline __host__ __device__ REAL2 operator* (const REAL2x2 &m, const REAL2 &rhs) {
			return make_REAL2(
			m.c[0]*rhs.x + m.c[2]*rhs.y,
			m.c[1]*rhs.x + m.c[3]*rhs.y);
}
