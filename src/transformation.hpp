#pragma once

#include "real.hpp"
#include "spline.hpp"
#include "vectors.hpp"
#include <iostream>

// Transform the mesh
struct Quaternion {
	REAL s;
	Vec3 v;
	Vec3 rotate(const Vec3 &point) const;
	static Quaternion from_axisangle(const Vec3 &axis, REAL angle);
	std::pair<Vec3, REAL> to_axisangle() const;
	Quaternion operator+(const Quaternion& q) const;
	Quaternion operator-(const Quaternion& q) const;
	Quaternion operator-() const;
	Quaternion operator*(const Quaternion& q) const;
	Quaternion operator*(REAL scalar) const;
	Quaternion operator/(REAL scalar) const;
};

Quaternion normalize(const Quaternion &q);
Quaternion inverse(const Quaternion &q);
REAL norm2(const Quaternion &q);
inline std::ostream &operator<< (std::ostream &out, const Quaternion &q) { out << "(" << q.s << ", " << q.v << ")"; return out; }

struct Transformation {
	Vec3 translation;
	REAL scale;
	Quaternion rotation;
	Transformation(REAL factor = 1);
	Vec3 apply(const Vec3 &point) const;
	Vec3 apply_vec(const Vec3 &vec) const;
	Transformation operator+(const Transformation& t) const;
	Transformation operator-(const Transformation& t) const;
	Transformation operator*(const Transformation& t) const;
	Transformation operator*(REAL scalar) const;
	Transformation operator/(REAL scalar) const;
};

Transformation identityT();
Transformation inverse(const Transformation &tr);
inline std::ostream &operator<< (std::ostream &out, const Transformation &t) { out << "(translation: " << t.translation << ", rotation: " << t.rotation << ", scale: " << t.scale << ")"; return out; }

typedef Spline<Transformation> Motion;
typedef std::pair<Transformation, Transformation> DTransformation;

void clean_up_quaternions(Motion &motion); // remove sign flips

Transformation get_trans(const Motion &motion, REAL t);
DTransformation get_dtrans(const Motion &motion, REAL t);
Vec3 apply_dtrans(const DTransformation &dT, const Vec3 &x0, Vec3 *vel = NULL);
Vec3 apply_dtrans_vec(const DTransformation &dT, const Vec3 &v0);
