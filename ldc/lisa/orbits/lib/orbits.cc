#include "orbits.hpp"


using namespace std;
using namespace PdbParam;

AnalyticOrbits::AnalyticOrbits(){}
AnalyticOrbits::AnalyticOrbits(double arm_length, double init_position, double init_rotation){

    for (int i = 0; i < number_of_spacecrafts; i += 1) {
      rotation[i] = i * inter_spacecraft_phase + init_rotation;
      c_rotation[i] = cos(rotation[i]);
      s_rotation[i] = sin(rotation[i]);
    }
    eccentricity = arm_length/(2*sqrt(3)*Nature::ASTRONOMICALUNIT_METER);
    init_time = init_position * ASTRONOMICAL_YEAR / (2 * M_PI);
  } 

double AnalyticOrbits::alpha(double time) {
  return BARYCENTER_ANGULAR_VELOCITY * (init_time + time);
} 
    
void AnalyticOrbits::position_x(int sci, double* time, double* x, int nt) {
    //vector<double> x(nt);
    for (int i = 0; i < nt; i += 1) {
      double a = alpha(time[i]);
      x[i] = Nature::ASTRONOMICALUNIT_METER * (cos(a) + eccentricity * (sin(a) * cos(a) * s_rotation[sci-1] - \
							 (1 + sin(a)*sin(a)) * c_rotation[sci-1]));
    }
}

void AnalyticOrbits::position_y(int sci, double* time, double* y, int nt) {
    //vector<double> x(nt);
    for (int i = 0; i < nt; i += 1) {
      double a = alpha(time[i]);
      y[i] = Nature::ASTRONOMICALUNIT_METER * (sin(a) + eccentricity * (sin(a) * cos(a) * c_rotation[sci-1] - \
						 (1 + cos(a)*cos(a)) * s_rotation[sci-1]));
    }
}

void AnalyticOrbits::position_z(int sci, double* time, double* z, int nt) {
    //vector<double> x(nt);
    for (int i = 0; i < nt; i += 1) {
      double a = alpha(time[i]);
      z[i] = -Nature::ASTRONOMICALUNIT_METER * eccentricity * sqrt(3) * cos(a-rotation[sci-1]);
    }
}

array<double, 3> AnalyticOrbits::position(int sci, double time) {
  //double x,y,z;
  array<double, 3> pos;
  position_x(sci, &time, &pos[0], 1) ;
  position_y(sci, &time, &pos[1], 1) ;
  position_z(sci, &time, &pos[2], 1) ;
  return pos;
}

void AnalyticOrbits::velocity_x(int sci, double* time, double* vx, int nt) {
  for (int i = 0; i < nt; i += 1) {
    double a = alpha(time[i]);
    double ca = cos(a);
    double sa = sin(a);
    vx[i] = Nature::ASTRONOMICALUNIT_METER * (-sa + eccentricity * ( (ca*ca-sa*sa) * s_rotation[sci-1] - \
					      2 * sa * ca * c_rotation[sci-1]) \
		       ) *  BARYCENTER_ANGULAR_VELOCITY;
  }
}
  
void AnalyticOrbits::velocity_y(int sci, double* time, double* vy, int nt) {
  for (int i = 0; i < nt; i += 1) {
    double a = alpha(time[i]);
    double ca = cos(a);
    double sa = sin(a);
    vy[i] = Nature::ASTRONOMICALUNIT_METER * (ca + eccentricity * ( (ca*ca-sa*sa) * c_rotation[sci-1] + \
					     2 * ca * sa * s_rotation[sci-1]) \
		       )*BARYCENTER_ANGULAR_VELOCITY;
  }
}

void AnalyticOrbits::velocity_z(int sci, double* time, double* vz, int nt) {
  for (int i = 0; i < nt; i += 1) {
    double a = alpha(time[i]);
    vz[i] = Nature::ASTRONOMICALUNIT_METER * eccentricity * sqrt(3) * sin(a-rotation[sci-1]) * BARYCENTER_ANGULAR_VELOCITY;
  }
}

array<double, 3> AnalyticOrbits::velocity(int sci, double time) {
  array<double, 3> vel;
  velocity_x(sci, &time, &vel[0], 1) ;
  velocity_y(sci, &time, &vel[1], 1) ;
  velocity_z(sci, &time, &vel[2], 1) ;
  return vel;
}

void AnalyticOrbits::get_travel_time(int emitter, int receiver, double* rec_time,
				     double* tt, int nt, int order){

  array<double, 3> v_j;
  array<double, 3> r_j;
  array<double, 3> r_i;
  
  for (int t = 0; t < nt; t += 1) {

    r_j = position(emitter, rec_time[t]);
    r_i = position(receiver, rec_time[t]);
    v_j = velocity(emitter, rec_time[t]);

    tt[t] = travel_time(r_i, r_j, v_j, order);
    
  }
}



