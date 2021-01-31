#pragma once

#include "obstacle.hpp"
#include "util.hpp"
#include "vectors.hpp"
#include <string>
#include <iostream>
#include <vector>

std::vector<Motion> load_mot(const std::string &filename, double fps);

// I think I'm going to ignore all the rest down here

typedef Vec<4> Vec4;

class mot_parser_exception {
public:
	mot_parser_exception(const std::string& error) : error(error) {}
	std::string error;
};

struct BodyFrame
{
	Vec3 pos;
	Vec4 orient;
};

typedef std::vector<BodyFrame> BodyFrameVector;
typedef std::vector<BodyFrameVector> BodyVector;

BodyVector read_motion_file(const std::string& filename);
BodyVector read_motion_file(std::istream& istr);

void write_motion_file(BodyVector& bodies, const std::string& filename);
void write_motion_file(BodyVector& bodies, std::ostream& ostr);

BodyFrame& get_body_frame(BodyVector& bodies, size_t body_index, size_t frame);
BodyFrameVector& get_body_frames(BodyVector& bodies, size_t body_index);
//BodyFrameVector& get_body_frames(size_t body_index);

std::vector<Spline<Transformation> > mot_to_spline(std::string motion_file,
	const Transformation& tr, double fps, double start_time, double pause_time);

std::vector<Obstacle> mot_to_obs(std::string motion_file,
	const Transformation& tr, std::string obj_basename, double fps,
	double start_time, double pause_time);

