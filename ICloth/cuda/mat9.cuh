typedef struct {
	REAL c[27];

	inline __host__ __device__ REAL3 m(int i) const { return make_REAL3(c[i*3], c[i*3+1], c[i*3+2]); }
} REAL3x9;

inline __host__ __device__ REAL3 getCol(const REAL3x9 &a, int i) {
	return make_REAL3(a.c[i*3], a.c[i*3+1], a.c[i*3+2]);
}

inline __host__ __device__ REAL3x9 getTrans (const REAL3x9 &m) {
	REAL3x9 t;

	getTrans(t.c, m.c);
	getTrans(t.c+9, m.c+9);
	getTrans(t.c+18, m.c+18);
	return t;
}

inline __host__ __device__ REAL3x3 operator *(const REAL3x9 &a, const REAL3 &b) {
	REAL3x3 t;

	for (int i = 0; i < 9; i++)
		t.c[i] = dot(a.m(i), b);
	return t;
}

typedef struct {
	REAL c[81];

	inline __host__ __device__ REAL &getIJ(int i, int j) { return c[j * 9 + i]; }
	inline __host__ __device__ REAL getIJ(int i, int j) const { return c[j * 9 + i]; }
} REAL9x9;

inline __host__ __device__ REAL9x9 operator- (const REAL9x9 &a)
{
	REAL9x9 t;

	for (int i=0; i<81; i++)
			t.c[i] = -a.c[i];

	return t;
}

inline __host__ __device__ REAL9x9 operator+ (const REAL9x9 &a, const REAL9x9 &b)
{
	REAL9x9 t;

	for (int i = 0; i<81; i++)
		t.c[i] = a.c[i]+b.c[i];

	return t;
}

inline __host__ __device__ void operator+= (REAL9x9 &a, const REAL9x9 &b)
{
	for (int i = 0; i<81; i++)
		a.c[i] += b.c[i];
}

inline __host__ __device__ REAL9x9 operator* (REAL a, const REAL9x9 &b)
{
	REAL9x9 t;

	for (int i = 0; i<81; i++)
		t.c[i] = a*b.c[i];
	
	return t;
}

inline __host__ __device__ REAL9x9 operator *(const REAL9x9 &b, REAL a) {
	REAL9x9 t;

	for (int i = 0; i<81; i++)
		t.c[i] = a*b.c[i];

	return t;
}

inline __host__ __device__ REAL9x9 operator *(const REAL3x9 &a, const REAL3x9 &b)
{
	REAL9x9 t;

	for (int i=0; i<9; i++)
		for (int j=0; j<9; j++) {
			t.getIJ(i, j) = dot(getCol(a, i), getCol(b, j));
		}
	return t;
}

typedef struct {
	REAL c[144];

	inline __host__ __device__ REAL &getIJ(int i, int j) { return c[j * 12 + i]; }
	inline __host__ __device__ REAL getIJ(int i, int j) const { return c[j * 12 + i]; }
} REAL12x12;

typedef struct {
	REAL c[12];
	inline __host__ __device__ REAL3 c0() const { return make_REAL3(c[0], c[1], c[2]); }
	inline __host__ __device__ REAL3 c1() const { return make_REAL3(c[3], c[4], c[5]); }
	inline __host__ __device__ REAL3 c2() const { return make_REAL3(c[6], c[7], c[8]); }
	inline __host__ __device__ REAL3 c3() const { return make_REAL3(c[9], c[10], c[11]); }
	inline __host__ __device__ REAL3 m(int i) const { return make_REAL3(c[i * 3], c[i * 3 + 1], c[i * 3 + 2]); }
} REAL3x4;

typedef REAL3x3 REAL9;
typedef REAL3x4 REAL12;

inline __host__ __device__ REAL9 operator *(const REAL9x9 &a, const REAL9 &b)
{
	REAL9 t;

	for (int i = 0; i < 9; i++) {
		t.c[i] = 0;

		for (int j = 0; j < 9; j++)
			t.c[i] += a.getIJ(i, j) * b.c[j];
	}

	return t;
}

inline __host__ __device__ REAL9x9 outer(const REAL9 &u, const REAL9 &v)
{
	REAL9x9 t;

	for (int j=0; j<9; j++)
		for (int i=0; i<9; i++) 
			t.getIJ(i, j) = getI(u, i)*getI(v, j);

	return t;
}


inline __host__ __device__ REAL3x4 make_REAL3x4
	(const REAL3 &a, const REAL3 &b, const REAL3 &c, const REAL3 &d)
{
	REAL3x4 t;

	t.c[0] = a.x; t.c[1] = a.y; t.c[2] = a.z;
	t.c[3] = b.x; t.c[4] = b.y; t.c[5] = b.z;
	t.c[6] = c.x; t.c[7] = c.y; t.c[8] = c.z;
	t.c[9] = d.x; t.c[10] = d.y; t.c[11] = d.z;

	return t;
}

inline __host__ __device__ REAL &getIJ(REAL12x12 &a, int i, int j)
{
	return a.c[j * 12 + i];
}

inline __host__ __device__ REAL getI(const REAL12 &a, int i)
{
	return a.c[i];
}

inline __host__ __device__ REAL12x12 outer(const REAL12 &u, const REAL12 &v)
{
	REAL12x12 t;

	for (int j=0; j<12; j++)
		for (int i=0; i<12; i++) 
			getIJ(t, i, j) = getI(u, i)*getI(v, j);

	return t;
}

inline __host__ __device__ REAL12x12 operator* (REAL a, const REAL12x12 &b) {
	REAL12x12 t;

	for (int i = 0; i < 144; i++)
		t.c[i] = a*b.c[i];

	return t;
}

inline __host__ __device__ REAL12x12 operator *(const REAL12x12 &b, REAL a) {
	REAL12x12 t;

	for (int i = 0; i < 144; i++)
		t.c[i] = a*b.c[i];

	return t;
}

inline __host__ __device__ REAL12 operator* (REAL a, const REAL12 &b) {
	REAL12 t;

	for (int i=0; i<12; i++)
		t.c[i] = a*b.c[i];
	
	return t;
}

inline __host__ __device__ REAL12 operator *(const REAL12 &b, REAL a) {
	REAL12 t;

	for (int i=0; i<12; i++)
		t.c[i] = a*b.c[i];
	
	return t;
}

inline __host__ __device__ REAL12x12 operator- (const REAL12x12 &a)
{
	REAL12x12 t;

	for (int i=0; i<144; i++)
		t.c[i] = -a.c[i];

	return t;
}


inline __host__ __device__ REAL12 operator *(const REAL12x12 &a, const REAL12 &b)
{
	REAL12 t;

	for (int i = 0; i < 12; i++) {
		t.c[i] = 0;
		for (int j = 0; j < 12; j++)
			t.c[i] += a.getIJ(i, j)*getI(b, j);
	}
	
	return t;
}

inline __host__ __device__ REAL12 operator +(const REAL12 &a, const REAL12 &b)
{
	REAL12 t;

	for (int i = 0; i < 12; i++)
		t.c[i] = a.c[i] + b.c[i];

	return t;
}
