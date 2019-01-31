/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 30;  // TODO: Set the number of particles
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x =  dist_x(gen);
    particle.y =  dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (Particle& p : particles) {
    double px;
	  double py;
    double ptheta;    
    
    if (fabs(yaw_rate) > 0.0001) {
      px = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      py = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      ptheta = p.theta + yaw_rate * delta_t;
    } else {
      px = p.x + velocity * delta_t * cos(p.theta);
      py = p.y+ velocity * delta_t * sin(p.theta);
      ptheta = p.theta + yaw_rate;
    }
    
    normal_distribution<double> dist_x(px, std_pos[0]);
    normal_distribution<double> dist_y(py, std_pos[1]);
    normal_distribution<double> dist_theta(ptheta, std_pos[2]);
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  } 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int pred_id = -1;
  double min_distance = numeric_limits<double>::max();
  for (LandmarkObs obs: observations) { 
    for (LandmarkObs pred: predicted) {

      double sqr_diff = dist(obs.x,obs.y,pred.x,pred.y);
      if ( (sqr_diff) < min_distance) {
        min_distance = sqr_diff;
        pred_id = pred.id;
      }
    }
    obs.id = pred_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double normalized_weight = 0.0;

  for (Particle& p : particles) {
    vector<LandmarkObs> trans_obs;
    for (size_t i=0; i <observations.size(); ++i) { 
      LandmarkObs trans_ob;
      trans_ob.id = i;
      trans_ob.x = p.x + (cos(p.theta) +observations[i].x) - (sin(p.theta) * observations[i].y);
      trans_ob.y = p.y + (sin(p.theta) +observations[i].x) + (cos(p.theta) * observations[i].y);
      trans_obs.push_back(trans_ob);
    }

    vector<LandmarkObs> filtered_landmarks;
    for (size_t j=0;j<map_landmarks.landmark_list.size();++j) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs(p.x - current_landmark.x_f) <= sensor_range) && (fabs(p.y - current_landmark.y_f) <= sensor_range) ){
        filtered_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }

    dataAssociation(filtered_landmarks, trans_obs);
    double gauss_norm, exponent;
    gauss_norm = 1 / (2 * M_PI *std_landmark[0]*std_landmark[1]);

    for (size_t k=0; k <trans_obs.size(); ++k) { 
      for (size_t l=0; l <filtered_landmarks.size(); ++l) { 
        if (trans_obs[k].id == filtered_landmarks[l].id) {
          exponent = exp (-1 *((pow((trans_obs[k].x-filtered_landmarks[l].x), 2)/(2 * pow(std_landmark[0],2))) + (pow((trans_obs[k].y-filtered_landmarks[l].y), 2)/(2 * pow(std_landmark[1],2)))));
          p.weight *= gauss_norm * exponent;
        }
      }
    }
  normalized_weight += p.weight;
  }
  for (size_t m=0; m<particles.size(); ++m) {
    particles[m].weight = particles[m].weight/normalized_weight;
    weights[m] = particles[m].weight;
  }  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::default_random_engine gen;

  double max_weight = 2.0 * *max_element(weights.begin(),weights.end());

  uniform_real_distribution<double> dist_weight(0.0, max_weight);
  uniform_int_distribution<int> dist_part(0, num_particles-1);

  double beta = 0.0;
  int index = dist_part(gen);

  vector<Particle> resampled_particles;
  for (int i = 0; i <num_particles; ++i) {
    beta += dist_weight(gen);
    while (beta > weights[index]) {
      beta-= weights[index];
      index = (index+1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
