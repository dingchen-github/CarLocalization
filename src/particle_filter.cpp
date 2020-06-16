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
using std::normal_distribution;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 20;  // TODO: Set the number of particles
    std::default_random_engine gen;
    // Create normal distributions
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    // Initialize every particle and append them to the list of particles
    for (int i = 0; i < num_particles; ++i){
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
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
    
    for (int i = 0; i < num_particles; ++i){
        double x, y, theta;
        if (fabs(yaw_rate) < 0.00001) {
            x = velocity * delta_t * cos(particles[i].theta);
            y = velocity * delta_t * sin(particles[i].theta);
            theta = 0;
        }
        else {
            x = velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            y = velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            theta = yaw_rate * delta_t;
        }
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
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
    // There are fewer observations than landmarks in the sensor range
    for (unsigned int i = 0; i < observations.size(); ++i){
        double distance = 1000.0;
        for (unsigned int j = 0; j < predicted.size(); ++j){
            double distance_new = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if (distance_new < distance){
                distance = distance_new;
                observations[i].id = predicted[j].id;
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
    for (int i = 0; i < num_particles; ++i){
        double x_part = particles[i].x;
        double y_part = particles[i].y;
        double theta_part = particles[i].theta;
        
        // Get landmarks within sensor range
        vector<LandmarkObs> predictions;
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j){
            int id = map_landmarks.landmark_list[j].id_i;
            double x = map_landmarks.landmark_list[j].x_f;
            double y = map_landmarks.landmark_list[j].y_f;
            if(fabs(x - x_part) <= sensor_range && fabs(y - y_part) <= sensor_range) {
                predictions.push_back(LandmarkObs{ id, x, y });
            }
        }
        
        // Coordination transformation for observations
        vector<LandmarkObs> observations_global;
        for (unsigned int j = 0; j < observations.size(); ++j){
            int id = 0; // Set all ids to 0, because they will be updated in dataAssociation
            double x_obs = observations[j].x;
            double y_obs = observations[j].y;
            double x = x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs);
            double y = y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs);
            observations_global.push_back(LandmarkObs{ id, x, y });
        }
        
        dataAssociation(predictions, observations_global);
        
        particles[i].weight = 1.0;
        double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        for (unsigned int j = 0; j < observations_global.size(); ++j){
            // map landmarks ids beginn with 1, so need to subtract 1 to get list index
            double mu_x = map_landmarks.landmark_list[observations_global[j].id-1].x_f;
            double mu_y = map_landmarks.landmark_list[observations_global[j].id-1].y_f;
            double exponent = (pow(observations_global[j].x - mu_x, 2) / (2 * pow(std_landmark[0], 2)))
            + (pow(observations_global[j].y - mu_y, 2) / (2 * pow(std_landmark[1], 2)));
            particles[i].weight *=  gauss_norm * exp(-exponent);
        }
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    
    // Get the list of weights
    weights.clear();
    for (int i = 0; i < num_particles; ++i){
        weights.push_back(particles[i].weight);
    }
    // double max_weight = *std::max_element(weights.begin(), weights.end());
    
    vector<Particle> p;
    std::default_random_engine gen;
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    for(int i = 0; i < num_particles; ++i) {
        p.push_back(particles[dist(gen)]);
    }
    //   double beta = 0.0;
    //   std::uniform_real_distribution<double> dis(0.0, 1.0);
    //   std::default_random_engine gen;
    //   int index = int(dis(gen) * num_particles);
    //   for (int i = 0; i < num_particles; ++i){
    //     beta += dis(gen) * 2.0 * max_weight;
    //     while(beta > weights[index]){
    //       beta -= weights[index];
    //       index = (index + 1) % num_particles;
    //     }
    //     p.push_back(particles[index]);
    //   }
    particles = p;
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