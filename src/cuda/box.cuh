
typedef struct __align__(16) _box3f {
	double3 _min, _max;

	inline __host__ __device__ void set(const double3 &a)
	{
		_min = _max = a;
	}

	inline __host__ __device__ void set(const double3 &a, const double3 &b)
	{
		_min = fminf(a, b);
		_max = fmaxf(a, b);
	}

	inline __host__ __device__  void set(const _box3f &a, const _box3f &b)
	{
		_min = fminf(a._min, b._min);
		_max = fmaxf(a._max, b._max);
	}

	inline __host__ __device__  void add(const double3 &a)
	{
		_min = fminf(_min, a);
		_max = fmaxf(_max, a);
	}

	inline __host__ __device__  void enlarge(double thickness)
	{
		_min -= make_double3(thickness);
		_max += make_double3(thickness);
	}

	inline __host__ __device__ bool overlaps(const _box3f& b) const
	{
		if (_min.x > b._max.x) return false;
		if (_min.y > b._max.y) return false;
		if (_min.z > b._max.z) return false;

		if (_max.x < b._min.x) return false;
		if (_max.y < b._min.y) return false;
		if (_max.z < b._min.z) return false;

		return true;
	}

	inline void print() {
		printf("%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	inline void print(FILE *fp) {
		fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}
} g_box;
