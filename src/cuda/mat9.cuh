typedef struct {
	double3x3 m[3];
} double3x9;

inline __host__ __device__ double3 getCol(const double3x9 &a, int i) {
	int k = i/3;
	return getCol(a.m[k], i-k*3);
}

inline __host__ __device__ double3x9 getTrans (const double3x9 &m) {
	double3x9 t;
	t.m[0] = getTrans(m.m[0]);
	t.m[1] = getTrans(m.m[1]);
	t.m[2] = getTrans(m.m[2]);
	return t;
}

inline __host__ __device__ double3x3 operator *(const double3x9 &a, const double3 &b) {
	double3x3 t;

	t.column0 = make_double3(dot(a.m[0].column0, b), dot(a.m[0].column1, b), dot(a.m[0].column2, b));
	t.column1 = make_double3(dot(a.m[1].column0, b), dot(a.m[1].column1, b), dot(a.m[1].column2, b));
	t.column2 = make_double3(dot(a.m[2].column0, b), dot(a.m[2].column1, b), dot(a.m[2].column2, b));
	return t;
}

typedef struct {
	double3x3 m[3][3];
} double9x9;

inline __host__ __device__ double &getIJ(double9x9 &a, int i, int j)
{
	int k = i/3;
	int l = j/3;

	return getIJ(a.m[k][l], i-k*3, j-l*3);
}

inline __host__ __device__ double9x9 make_double9x9 (const double3x3 v[])
{
	double9x9 t;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			t.m[i][j] = v[i*3+j];

	return t;
}

inline __host__ __device__ double9x9 operator- (const double9x9 &a)
{
	double9x9 t;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			t.m[i][j] = -a.m[i][j];

	return t;
}

inline __host__ __device__ double9x9 operator+ (const double9x9 &a, const double9x9 &b)
{
	double9x9 t;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			t.m[i][j] = a.m[i][j]+b.m[i][j];
	
	return t;
}

inline __host__ __device__ void operator+= (double9x9 &a, const double9x9 &b)
{
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			a.m[i][j] += b.m[i][j];
}

inline __host__ __device__ double9x9 operator* (double a, const double9x9 &b)
{
	double9x9 t;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			t.m[i][j] = a*b.m[i][j];
	
	return t;
}

inline __host__ __device__ double9x9 operator *(const double9x9 &b, double a) {
	double9x9 t;

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			t.m[i][j] = a*b.m[i][j];
	
	return t;
}

inline __host__ __device__ double9x9 operator *(const double3x9 &a, const double3x9 &b)
{
	double9x9 t;

	for (int i=0; i<9; i++)
		for (int j=0; j<9; j++) {
			getIJ(t, i, j) = dot(getCol(a, i), getCol(b, j));
		}
	return t;
}

typedef struct {
	double3x3 m[4][4];
} double12x12;

typedef struct {
	double3 m[4];
} double3x4;

typedef double3x3 double9;
typedef double3x4 double12;

inline __host__ __device__ double9 operator *(const double9x9 &a, const double9 &b)
{
	double9 t;

	t.column0 = a.m[0][0]*b.column0+a.m[0][1]*b.column1+a.m[0][2]*b.column2;
	t.column1 = a.m[1][0]*b.column0+a.m[1][1]*b.column1+a.m[1][2]*b.column2;
	t.column2 = a.m[2][0]*b.column0+a.m[2][1]*b.column1+a.m[2][2]*b.column2;
	return t;
}

inline __host__ __device__ double9x9 outer(const double9 &u, const double9 &v)
{
	double9x9 t;

	for (int j=0; j<9; j++)
		for (int i=0; i<9; i++) 
			getIJ(t, i, j) = getI(u, i)*getI(v, j);

	return t;
}


inline __host__ __device__ double3x4 make_double3x4
	(const double3 &a, const double3 &b, const double3 &c, const double3 &d)
{
	double3x4 t;

	t.m[0] = a;
	t.m[1] = b;
	t.m[2] = c;
	t.m[3] = d;

	return t;
}

inline __host__ __device__ double &getIJ(double12x12 &a, int i, int j)
{
	int k = i/3;
	int l = j/3;

	return getIJ(a.m[k][l], i-k*3, j-l*3);
}

inline __host__ __device__ double getI(const double12 &a, int i)
{
	int id = i/3;

	return getI(a.m[id], i-id*3);
}

inline __host__ __device__ double12x12 outer(const double12 &u, const double12 &v)
{
	double12x12 t;

	for (int j=0; j<12; j++)
		for (int i=0; i<12; i++) 
			getIJ(t, i, j) = getI(u, i)*getI(v, j);

	return t;
}

inline __host__ __device__ double12x12 operator* (double a, const double12x12 &b) {
	double12x12 t;

	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
			t.m[i][j] = a*b.m[i][j];
	
	return t;
}

inline __host__ __device__ double12x12 operator *(const double12x12 &b, double a) {
	double12x12 t;

	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
			t.m[i][j] = a*b.m[i][j];
	
	return t;
}

inline __host__ __device__ double12 operator* (double a, const double12 &b) {
	double12 t;

	for (int i=0; i<4; i++)
		t.m[i] = a*b.m[i];
	
	return t;
}

inline __host__ __device__ double12 operator *(const double12 &b, double a) {
	double12 t;

	for (int i=0; i<4; i++)
		t.m[i] = a*b.m[i];
	
	return t;
}

inline __host__ __device__ double12x12 operator- (const double12x12 &a)
{
	double12x12 t;

	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
			t.m[i][j] = -a.m[i][j];

	return t;
}


inline __host__ __device__ double12 operator *(const double12x12 &a, const double12 &b)
{
	double12 t;

	t.m[0] = a.m[0][0]*b.m[0]+a.m[0][1]*b.m[1]+a.m[0][2]*b.m[2]+a.m[0][3]*b.m[3];
	t.m[1] = a.m[1][0]*b.m[0]+a.m[1][1]*b.m[1]+a.m[1][2]*b.m[2]+a.m[1][3]*b.m[3];
	t.m[2] = a.m[2][0]*b.m[0]+a.m[2][1]*b.m[1]+a.m[2][2]*b.m[2]+a.m[2][3]*b.m[3];
	t.m[3] = a.m[3][0]*b.m[0]+a.m[3][1]*b.m[1]+a.m[3][2]*b.m[2]+a.m[3][3]*b.m[3];
	return t;
}

inline __host__ __device__ double12 operator +(const double12 &a, const double12 &b)
{
	double12 t;

	for (int i=0; i<4; i++)
		t.m[i] = a.m[i]+b.m[i];

	return t;
}

