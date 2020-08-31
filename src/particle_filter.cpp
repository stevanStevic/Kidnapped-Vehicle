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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 200;  // TODO: Set the number of particles

  // Get standard deviations for Gaussian (Normal) distribution
  const double std_dev_x = std[0];
  const double std_dev_y = std[1];
  const double std_dev_theta = std[2];

  std::default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y, theta
  std::normal_distribution<double> dist_x(x, std_dev_x);
  std::normal_distribution<double> dist_y(y, std_dev_y);
  std::normal_distribution<double> dist_theta(theta, std_dev_theta);

  for(auto i = 0; i < num_particles; ++i)
  {
    // Sample from normal distributions for every particle
    const double sample_x = dist_x(gen);
    const double sample_y = dist_y(gen);
    const double sample_theta = dist_theta(gen);

    const double weight{1.0};

    Particle particle{i, sample_x, sample_y, sample_theta, weight};
    particles.push_back(particle);
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
  // Get standard deviations for Gaussian (Normal) distribution
  const double std_dev_x = std_pos[0];
  const double std_dev_y = std_pos[1];
  const double std_dev_theta = std_pos[2];

  std::default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y, theta
  std::normal_distribution<double> dist_x(0.0, std_dev_x);
  std::normal_distribution<double> dist_y(0.0, std_dev_y);
  std::normal_distribution<double> dist_theta(0.0, std_dev_theta);

  for(auto& particle : particles)
  {
    const auto x0{particle.x};
    const auto y0{particle.y};
    const auto theta0{particle.theta};

    // Different formulas are used if we are driving straight and changing steering angle
    if(std::fabs(yaw_rate) > 0.001)
    {
      const double yaw_change{yaw_rate * delta_t};
      const double vel_div_theta_dot{velocity / yaw_rate};

      // xf​=x0​ + v/θ˙​ * [sin(θ0​+θ˙(dt)) − sin(θ0​)]
      particle.x = x0 + (vel_div_theta_dot * (std::sin(theta0 + yaw_change) - std::sin(theta0)));

      // yf​=y0​ + v​/θ˙ * [cos(θ0​) − cos(θ0​+θ˙(dt))]
      particle.y = y0 + (vel_div_theta_dot * (std::cos(theta0) - std::cos(theta0 + yaw_change)));

      // θf​=θ0​+θ˙(dt)
      particle.theta = theta0 + yaw_change;
    }
    else
    {
      const double displacment{velocity * delta_t};

      // xf = x0 + v*dt cos(θ0​)
      particle.x = x0 + (displacment * std::cos(theta0));

      // yf​ = y0​ + v*dt * sin(θ0​)
      particle.y = y0 + (displacment * std::sin(theta0));

      // θf​= θ0
      // Nothing to be done, theta stays the same
    }

    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta +=  dist_theta(gen);
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
  for(auto& observation : observations)
  {
    double min_distance{std::numeric_limits<double>::infinity()};

    for(const auto& landmark : predicted)
    {
      const double distance = dist(observation.x, observation.y, landmark.x, landmark.y);

      if (distance < min_distance)
      {
        observation = landmark;
      }
    }
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

  //Observation vehicle frame transformation to map frame according to each particle
  for(auto& particle : particles)
  {
    double weight{1.0};
    const double x_particle{particle.x};
    const double y_particle{particle.y};

    std::vector<LandmarkObs> landmarks_in_range;
    for (const auto& map_landmark : map_landmarks.landmark_list)
    {
      const double dist_to_landmark = dist(x_particle, y_particle, map_landmark.x_f, map_landmark.y_f);

      if (dist_to_landmark < sensor_range)
      {
        landmarks_in_range.push_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
      }
    }

    for (const auto& observation: observations)
    {
      std::vector<LandmarkObs> transformed_landmarks;

      const double theta_particle{particle.theta};
      const double x_obervation{observation.x};
      const double y_obervation{observation.y};

      // transform to map x coordinate
      // xm​=xp​+(cosθ×xc​)−(sinθ×yc​)
      const double x_map = x_particle + (std::cos(theta_particle) * x_obervation) - (std::sin(theta_particle) * y_obervation);

      // transform to map y coordinate
      // ym​=yp​+(sinθ×xc​)+(cosθ×yc​)
      const double y_map = y_particle + (std::sin(theta_particle) * x_obervation) + (std::cos(theta_particle) * y_obervation);

      double min_distance{std::numeric_limits<double>::max()};
      LandmarkObs closest_landmark;
      for(const auto& landmark : landmarks_in_range)
      {
        const double distance = dist(observation.x, observation.y, landmark.x, landmark.y);

        if (distance < min_distance)
        {
          min_distance = distance;
          closest_landmark = landmark;
        }
      }

      // const auto closet_landmark = landmarks_in_range[closest_landmark_index];
      weight *= multiv_prob(std_landmark[0], std_landmark[1], x_map, y_map, closest_landmark.x, closest_landmark.y);
    }

    particle.weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> weights;

  for(const auto& particle : particles)
  {
    weights.push_back(particle.weight);
  }

  std::default_random_engine gen;
  std::discrete_distribution<> discrete_dist(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;
  for(auto i = 0; i < num_particles; ++i)
  {
    const auto new_particle = particles[discrete_dist(gen)];
    resampled_particles.push_back(new_particle);
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