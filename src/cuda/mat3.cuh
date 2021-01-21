typedef struct {
	double3 column0;
	double3 column1;
} double3x2;

typedef struct {
	double2 column0;
	double2 column1;
	double2 column2;
} double2x3;

typedef struct {
	double2 column0;
	double2 column1;
} double2x2;

inline __host__ __device__ double3x2 make_double3x2(double3 c0, double3 c1)
{
	double3x2 t;

	t.column0 = c0;
	t.column1 = c1;
	return t;
}

inline __host__ __device__ double2x3 make_double2x3(double2 c0, double2 c1, double2 c2)
{
	double2x3 t;

	t.column0 = c0;
	t.column1 = c1;
	t.column2 = c2;
	return t;
}

inline __host__ __device__ double2x2 make_double2x2(double2 c0, double2 c1)
{
	double2x2 t;

	t.column0 = c0;
	t.column1 = c1;
	return t;
}

inline __host__ __device__ double det(double2x2 t)
{
	return t.column0.x * t.column1.y - t.column0.y * t.column1.x;
}

inline __host__ __device__ double2x2 inverse(const double2x2 &t)
{
	double2x2 r = make_double2x2 (
		make_double2( t.column1.y,  -t.column0.y),
		make_double2( -t.column1.x,  t.column0.x));

	double detInv = 1.0f/det(t);

	r.column0 *= detInv;
	r.column1 *= detInv;

	return r;
}

inline __host__ __device__ double2x3 operator* (const double2x2 &a, const double2x3 &b)
{
	return make_double2x3(
		make_double2(
		a.column0.x*b.column0.x+a.column1.x*b.column0.y,
		a.column0.y*b.column0.x+a.column1.y*b.column0.y),
		make_double2(
		a.column0.x*b.column1.x+a.column1.x*b.column1.y,
		a.column0.y*b.column1.x+a.column1.y*b.column1.y),
		make_double2(
		a.column0.x*b.column2.x+a.column1.x*b.column2.y,
		a.column0.y*b.column2.x+a.column1.y*b.column2.y)
	);
}

inline __host__ __device__ double3x2 operator* (const double3x2 &a, const double2x2 &b)
{
	return make_double3x2(
		make_double3(
		a.column0.x*b.column0.x+a.column1.x*b.column0.y,
		a.column0.y*b.column0.x+a.column1.y*b.column0.y,
		a.column0.z*b.column0.x+a.column1.z*b.column0.y),
		make_double3(
		a.column0.x*b.column1.x+a.column1.x*b.column1.y,
		a.column0.y*b.column1.x+a.column1.y*b.column1.y,
		a.column0.z*b.column1.x+a.column1.z*b.column1.y)
	);
}

inline __host__ __device__ double3 operator* (const double3x2 &a, const double2 &b)
{
	return make_double3(
		a.column0.x * b.x + a.column1.x * b.y,
		a.column0.y * b.x + a.column1.y * b.y,
		a.column0.z * b.x + a.column1.z * b.y);
}

/////////////////////////////////////////////////////////////////////////////////////////////
typedef struct {
	double3 column0;
	double3 column1;
	double3 column2;
} double3x3;

typedef struct {
    double3x3 u;
    double2 s;
    double2x2 vt;
} svd3x2;

inline __host__ __device__ double3 getCol(double3x3 a, int i)
{
	if (i == 0)
		return a.column0;
	else if (i == 1)
		return a.column1;
	else
		return a.column2;
}

inline __host__ __device__ double3x3 make_double3x3 (double3 c0, double3 c1, double3 c2) {
	double3x3 t;
	t.column0 = c0;
	t.column1 = c1;
	t.column2 = c2;
	return t;
}

inline __host__ __device__ double getIJ(const double3x3 &m, int i, int j)
{
	if (j == 0) {
		if (i == 0)
			return m.column0.x;
		else if (i == 1)
			return m.column0.y;
		else if (i == 2)
			return m.column0.z;
	} else if (j == 1) {
		if (i == 0)
			return m.column1.x;
		else if (i == 1)
			return m.column1.y;
		else if (i == 2)
			return m.column1.z;
	} else if (j == 2) {
		if (i == 0)
			return m.column2.x;
		else if (i == 1)
			return m.column2.y;
		else if (i == 2)
			return m.column2.z;
	}

	assert(0);
	return 0;
}

inline __host__ __device__ double &getIJ(double3x3 &m, int i, int j)
{
	if (j == 0) {
		if (i == 0)
			return m.column0.x;
		else if (i == 1)
			return m.column0.y;
		else if (i == 2)
			return m.column0.z;
	} else if (j == 1) {
		if (i == 0)
			return m.column1.x;
		else if (i == 1)
			return m.column1.y;
		else if (i == 2)
			return m.column1.z;
	} else if (j == 2) {
		if (i == 0)
			return m.column2.x;
		else if (i == 1)
			return m.column2.y;
		else if (i == 2)
			return m.column2.z;
	}

	assert(0);
	return m.column0.x;
}

inline __host__ __device__ double getIJ(const double2x2 &m, int i, int j)
{
	if (j == 0) {
		if (i == 0)
			return m.column0.x;
		else if (i == 1)
			return m.column0.y;
	} else if (j == 1) {
		if (i == 0)
			return m.column1.x;
		else if (i == 1)
			return m.column1.y;
	} 

	assert(0);
	return 0;
}

inline __host__ __device__ double getI(const double2x2 &m, int i)
{
	if (i == 0)
		return m.column0.x;
	if (i == 1)
		return m.column0.y;
	if (i == 2)
		return m.column1.x;
	if (i == 3)
		return m.column1.y;

	assert(0);
	return 0;
}

inline __host__ __device__ double &getI(double2x2 &m, int i)
{
	if (i == 0)
		return m.column0.x;
	if (i == 1)
		return m.column0.y;
	if (i == 2)
		return m.column1.x;
	if (i == 3)
		return m.column1.y;

	assert(0);
	return m.column0.x;
}

inline __host__ __device__ double getI(const double3x3 &m, int i)
{
	int id = i/3;
	return getI(getCol(m, id), i-id*3);
}

inline __host__ void print_double3x3 (double3x3 * m) {
	printf("%f ", m->column0.x); printf("%f ", m->column1.x); printf("%f \n", m->column2.x); 
	printf("%f ", m->column0.y); printf("%f ", m->column1.y); printf("%f \n", m->column2.y); 
	printf("%f ", m->column0.z); printf("%f ", m->column1.z); printf("%f \n\n", m->column2.z); 
}

inline __host__ __device__ double3x3 operator+ (const double3x3 &m1, const double3x3 &m2)
{
	return make_double3x3(m1.column0 + m2.column0,
						m1.column1 + m2.column1,
						m1.column2 + m2.column2);
}

inline __host__ __device__ void operator+= (double3x3 &m1, const double3x3 &m2)
{
	m1.column0 += m2.column0;
	m1.column1 += m2.column1;
	m1.column2 += m2.column2;
}


inline __host__ __device__ double3x3 operator- (const double3x3 &m1, const double3x3 &m2) {
	return make_double3x3(m1.column0 - m2.column0,
						m1.column1 - m2.column1,
						m1.column2 - m2.column2);
}

inline __host__ __device__ double2x2 operator- (const double2x2 &m1, const double2x2 &m2) {
	return make_double2x2(m1.column0 - m2.column0,
						m1.column1 - m2.column1);
}

inline __host__ __device__ double3x3 operator- (const double3x3 &a) {
		return make_double3x3(-a.column0, -a.column1, -a.column2);
	}


inline __host__ __device__ double3x3 operator* (double a, const double3x3 &m) {
	return make_double3x3(a * m.column0, a * m.column1, a * m.column2);
}

inline __host__ __device__ double3x3 operator* (const double3x3 &m, double a) {
	return make_double3x3(a * m.column0, a * m.column1, a * m.column2);
}

inline __host__ __device__ void operator*= (double3x3 &m, double a) {
	m.column0 *= a;
	m.column1 *= a;
	m.column2 *= a;
}

inline __host__ __device__ void operator*= (double2x2 &m, double a) {
	m.column0 *= a;
	m.column1 *= a;
}

inline __host__ __device__ double2x2 operator* (double a, const double2x2 &m) {
	return make_double2x2(a * m.column0, a * m.column1);
}

inline __host__ __device__ double2x2 operator* (const double2x2 &m, double a) {
	return make_double2x2(a * m.column0, a * m.column1);
}

inline __host__ __device__ double3x3 getTrans (const double3x3 &m) {
	double3x3 t;
	t.column0 = make_double3(m.column0.x, m.column1.x, m.column2.x);
	t.column1 = make_double3(m.column0.y, m.column1.y, m.column2.y);
	t.column2 = make_double3(m.column0.z, m.column1.z, m.column2.z);
	return t;
}

inline __host__ __device__ double2x3 getTrans (const double3x2 &m) {
	double2x3 t;
	t.column0 = make_double2(m.column0.x, m.column1.x);
	t.column1 = make_double2(m.column0.y, m.column1.y);
	t.column2 = make_double2(m.column0.z, m.column1.z);
	return t;
}

inline __host__ __device__ double3x2 getTrans (const double2x3 &m) {
	double3x2 t;
	t.column0 = make_double3(m.column0.x, m.column1.x, m.column2.x);
	t.column1 = make_double3(m.column0.y, m.column1.y, m.column2.y);
	return t;
}

inline __host__ __device__ double2x2 getTrans (const double2x2 &m) {
	double2x2 t;
	t.column0 = make_double2(m.column0.x, m.column1.x);
	t.column1 = make_double2(m.column0.y, m.column1.y);
	return t;
}

inline __host__ __device__ double3x3 operator* (const double3x3 &m1, const double3x3 &m2) {
	double3x3 m1T = getTrans(m1);

	return make_double3x3(
		make_double3(dot(m1T.column0, m2.column0), dot(m1T.column1, m2.column0), dot(m1T.column2, m2.column0)),
		make_double3(dot(m1T.column0, m2.column1), dot(m1T.column1, m2.column1), dot(m1T.column2, m2.column1)),
		make_double3(dot(m1T.column0, m2.column2), dot(m1T.column1, m2.column2), dot(m1T.column2, m2.column2)));
}

inline __host__ __device__ double3x2 operator* (const double3x3 &m1, const double3x2 &m2)
{
	double3x3 m1T = getTrans(m1);
	return make_double3x2(
		make_double3(dot(m1T.column0, m2.column0), dot(m1T.column1, m2.column0), dot(m1T.column2, m2.column0)),
		make_double3(dot(m1T.column0, m2.column1), dot(m1T.column1, m2.column1), dot(m1T.column2, m2.column1)));
}

inline __host__ __device__ double2x2 operator* (const double2x3 &m1, const double3x2 &m2) {
	double3x2 m1T = getTrans(m1);

	return make_double2x2(
		make_double2(dot(m1T.column0, m2.column0), dot(m1T.column1, m2.column0)),
		make_double2(dot(m1T.column0, m2.column1), dot(m1T.column1, m2.column1)));
}

inline __host__ __device__ double3x3 operator* (const double3x2 &m1, const double2x3 &m2) {
	double2x3 m1T = getTrans(m1);

	return make_double3x3(
		make_double3(dot(m1T.column0, m2.column0), dot(m1T.column1, m2.column0), dot(m1T.column2, m2.column0)),
		make_double3(dot(m1T.column0, m2.column1), dot(m1T.column1, m2.column1), dot(m1T.column2, m2.column1)),
		make_double3(dot(m1T.column0, m2.column2), dot(m1T.column1, m2.column2), dot(m1T.column2, m2.column2)));
}

inline __host__ __device__ double3x3 outer(const double3 &a, const double3 &b){
	return make_double3x3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ double3x3 identity3x3(){
	return make_double3x3(
					   make_double3(1.0, 0.0, 0.0),
					   make_double3(0.0, 1.0, 0.0),
					   make_double3(0.0, 0.0, 1.0));
}

inline __host__ __device__ double2x2 identity2x2(){
	return make_double2x2(
					   make_double2(1.0, 0.0),
					   make_double2(0.0, 1.0));
}

inline __host__ __device__ double3x3 make_double3x3(double m){
	return make_double3x3(
					   make_double3(m, 0.0, 0.0),
					   make_double3(0.0, m, 0.0),
					   make_double3(0.0, 0.0, m)
					   );
}

inline __host__ __device__ double2x2 make_double2x2(double m){
	return make_double2x2(
					   make_double2(m, 0.0),
					   make_double2(0.0, m)
					   );
}

inline __host__ __device__ double3x3 zero3x3(){
	return make_double3x3(
					   make_double3(0.0, 0.0, 0.0),
					   make_double3(0.0, 0.0, 0.0),
					   make_double3(0.0, 0.0, 0.0)
					   );
}

inline __host__ __device__ double2x2 zero2x2()
{
	return make_double2x2(
		make_double2(0, 0),
		make_double2(0, 0));
}

inline __host__ __device__ double3 getRow(double2x3 m, int i)
{
	if (i == 0)
		return make_double3(m.column0.x, m.column1.x, m.column2.x);
	if (i == 1)
		return make_double3(m.column0.y, m.column1.y, m.column2.y);

	assert(0);
	return zero3f();
}

inline __host__ __device__ double3x3 getInverse (double3x3 &m) {
	double3x3 t;
	t.column0.x = m.column1.y*m.column2.z - m.column1.z*m.column2.y;
	t.column0.y = m.column2.y*m.column0.z - m.column2.z*m.column0.y;
	t.column0.z = m.column0.y*m.column1.z - m.column0.z*m.column1.y;

	t.column1.x = m.column1.z*m.column2.x - m.column1.x*m.column2.z;
	t.column1.y = m.column2.z*m.column0.x - m.column2.x*m.column0.z;
	t.column1.z = m.column0.z*m.column1.x - m.column0.x*m.column1.z;

	t.column2.x = m.column1.x*m.column2.y - m.column1.y*m.column2.x;
	t.column2.y = m.column2.x*m.column0.y - m.column2.y*m.column0.x;
	t.column2.z = m.column0.x*m.column1.y - m.column0.y*m.column1.x;

	double det = m.column0.x*t.column0.x + m.column0.y*t.column1.x + m.column0.z*t.column2.x;
	double detInv = 1.0f/det;

	t.column0*=detInv;
	t.column1*=detInv;
	t.column2*=detInv;

	return t;
}

inline __host__ __device__ double3x3 rotation(double3 &axis, double theta ) {
	double s = sin( theta );
	double c = cos( theta );
	double t = 1-c;
	double x = axis.x, y = axis.y, z = axis.z;
		
/*	_data[0] = t*x*x + c;
	_data[1] = t*x*y - s*z;
	_data[2] = t*x*z + s*y;
	_data[3] = t*x*y + s*z;
	_data[4] = t*y*y + c;
	_data[5] = t*y*z - s*x;
	_data[6] = t*x*z - s*y;
	_data[7] = t*y*z + s*x;
	_data[8] = t*z*z + c;*/

	double3x3 tt;
	tt.column0.x = t*x*x + c;
	tt.column1.x = t*x*y - s*z;
	tt.column2.x = t*x*z + s*y;
	tt.column0.y= t*x*y + s*z;
	tt.column1.y = t*y*y + c;
	tt.column2.y = t*y*z - s*x;
	tt.column0.z = t*x*z - s*y;
	tt.column1.z = t*y*z + s*x;
	tt.column2.z = t*z*z + c;

	return tt;
}
	
inline __host__ __device__ double3 operator* (const double3x3 &m, const double3 &rhs) {
			return make_double3(
			m.column0.x*rhs.x + m.column1.x*rhs.y + m.column2.x*rhs.z,
			m.column0.y*rhs.x + m.column1.y*rhs.y + m.column2.y*rhs.z,
			m.column0.z*rhs.x + m.column1.z*rhs.y + m.column2.z*rhs.z);
}

inline __host__ __device__ double2 operator* (const double2x2 &m, const double2 &rhs) {
			return make_double2(
			m.column0.x*rhs.x + m.column1.x*rhs.y,
			m.column0.y*rhs.x + m.column1.y*rhs.y);
}

inline __host__ __device__ void
singular_value_decomposition (double3x2 &A, svd3x2 &svd)
{
	double3 c0 = A.column0;
	double3 c1 = A.column1;
	double a0   = dot(c0, c0); //				|a0  b|
	double b    = dot(c0, c1);  // A*A' =	|		  |
	double a1   = dot(c1, c1); //				|b  a1|
	double am   = a0 - a1;
	double ap   = a0 + a1;
	double det  = sqrt(am*am + 4*b*b);

	// eigen values
	double ev0  = sqrt(0.5 * (ap + det));
	double ev1  = sqrt(0.5 * (ap - det));
	svd.s = make_double2(ev0, ev1);

	// http://en.wikipedia.org/wiki/Trigonometric_identities
	double sina, cosa;
	if (b == 0) {
		sina = 0;
		cosa = 1;
	} else {
		double tana = (am - det) / (2*b);
		cosa = 1.0/sqrt(1 + tana*tana);
		sina = tana * cosa;
	}

  // 2x2
	svd.vt = make_double2x2(
		make_double2(-cosa, sina),
		make_double2(sina, cosa));

  // 3x3
	double t00  = -cosa/ev0;
	double t10  =  sina/ev0;
	double3 uc0 = make_double3(t00*c0.x + t10*c1.x, t00*c0.y + t10*c1.y, t00*c0.z + t10*c1.z);

	double t01  =  sina/ev1;
	double t11  =  cosa/ev1;
	double3 uc1 = make_double3(t01*c0.x + t11*c1.x, t01*c0.y + t11*c1.y, t01*c0.z + t11*c1.z);

	double3 uc2 = cross(uc0, uc1);
	svd.u = make_double3x3(uc0, uc1, uc2);
}
