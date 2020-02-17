#!/usr/bin/env python3
#  This is a visualization tool designed to aid the rover domain imaging

# Modified by Enna Sachdeva
# Date: 1/19/2020

# Modified by Ashwin Vinoo
# Date: 2/7/2019

# Created by Nick.

# import all the necessary modules
import numpy as np
import random
import pygame
import time
import math
import os

# ----------- Hyper Parameters -----------

primary_caption = 'MERL in rover domain' # The main caption displayed over the pygame screen
text_at_bottom = 'Rover incentivized to go to UAVs, UAVs incentivized to go to POI' # The text to display at the bottom of the screen
pygame_font_size = 40 # This is the font size of the text shown over the pygame display
pygame_font = 'Times New Roman' # This is the font shown over the pygame display
background_image_location = 'images/background.png' # The background image file name to use

# Files to load for the images of the visualizer
rover_image_location = 'images/rover.png'
uav_image_location = 'images/uav.png'
green_poi_image_location = 'images/greenflag.png'
red_poi_image_location = 'images/redflag.png'
observation_ring_image_location = 'observation_ring_8.png' #todo: to be done
# This will allow you to disable the observation ring display
display_rings = False #todo: to be done

# scale of rovers and POIs w.r.t window
rover_image_scale = 0.07 # The ratio between the rover height and the the average of the window height and width
uav_image_scale = 0.07 # The ratio between the rover height and the the average of the window height and width
poi_image_scale = 0.04 # The ratio between the rover height and the the average of the window height and width

# window size
window_width = 1080
window_height = 1080

# some offset settings
window_width_offset = 78 # The displacement of the pygame window from the left border of the screen
window_height_offset = 0 # The displacement of the pygame window from the upper border of the screen


#todo: are these required
circle_radius_ratio = 0.003 # The ratio between the rover trail circle radius and the average of the window height and width
line_width_ratio = 0.003 # The ratio between the rover trail line width and the average of the window height and width


#todo: below one is not required
# The background image file location is appended with the folder locations
# -----------------------------------------
'''
background_image_location = './display_images/background/' + background_image_location
rover_image_location = './display_images/rovers/' + rover_image_location
green_poi_image_location = './display_images/poi/' + green_poi_image_location
red_poi_image_location = './display_images/poi/' + red_poi_image_location
observation_ring_image_location = './display_images/observation_ring/' + observation_ring_image_location
'''

# The rover domain visualizer
class RoverDomainVisualizer(object):

    '''
    # This function helps visualize the rover domain environment
    # resolution: the size of the pygame window
    # window offset: allows us to move the pygame window to the right or downwards (makes space for the task bar access)
    '''

    def __init__(self, rover_count, uav_count, grid_size, observation_radius, coupling_factor_uav, coupling_factor_rover,
                 resolution=(window_width, window_height), window_offset=(window_width_offset, window_height_offset)):

        # The pygame window will be initialized with these displacements from the border of the screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(window_offset[0]) + "," + str(window_offset[1])
        pygame.init() # Initialize all imported pygame modules

        self.pygame_font = pygame.font.SysFont(pygame_font, pygame_font_size) # Load a new font from a given filename or a python file object. The size is the height of the font in pixels
        self.pygame_display = pygame.display.set_mode(resolution, pygame.NOFRAME)  # Sets the pygame display to cover the maximum space permitted as we are ignoring the resolution argument

        pygame.display.set_caption(primary_caption) # Sets the caption to be displayed on the window
        display_info = pygame.display.Info() # Get the current pygame window information

        # pygame window dimensions
        self.display_width = display_info.current_w
        self.display_height = display_info.current_h

        # grid size used for rover domain learning algorithm
        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]

        # The coupling factor used in rover domain
        self.coupling_factor_uav = coupling_factor_uav
        self.coupling_factor_rover = coupling_factor_rover

        # Text surface dimensions
        self.text_surface = self.pygame_font.render(text_at_bottom, True, (0, 0, 0))
        self.text_surface_width = self.text_surface.get_width()
        self.text_surface_height = self.text_surface.get_height()

        # Loads the images
        self.background_image = pygame.image.load(background_image_location)
        self.rover_image = pygame.image.load(rover_image_location)
        self.uav_image = pygame.image.load(uav_image_location)
        self.green_poi_image = pygame.image.load(green_poi_image_location)
        self.red_poi_image = pygame.image.load(red_poi_image_location)
        #self.observation_ring_image = pygame.image.load(observation_ring_image_location) #todo: need to see

        # Images dimensions
        rover_image_width = self.rover_image.get_width()
        rover_image_height = self.rover_image.get_height()
        uav_image_width = self.uav_image.get_width()
        uav_image_height = self.uav_image.get_height()
        green_poi_image_width = self.green_poi_image.get_width()
        green_poi_image_height = self.green_poi_image.get_height()
        red_poi_image_width = self.red_poi_image.get_width()
        red_poi_image_height = self.red_poi_image.get_height()

        # average dimensions of display #todo: why its needed?
        average_display_dimension = (self.display_width + self.display_height)/2
        average_rover_dimension = (rover_image_width + rover_image_height)/2
        average_uav_dimension = (uav_image_width + uav_image_height)/2
        average_green_poi_dimension = (green_poi_image_width + green_poi_image_height)/2
        average_red_poi_dimension = (red_poi_image_width + red_poi_image_height)/2

        # The scaled dimensions of the rover image
        self.rover_width = int((rover_image_scale*average_display_dimension) *
                               (rover_image_width/average_rover_dimension))
        self.rover_height = int((rover_image_scale*average_display_dimension) *
                                (rover_image_height/average_rover_dimension))

        self.uav_width = int((uav_image_scale * average_display_dimension) *
                               (uav_image_width / average_rover_dimension))
        self.uav_height = int((uav_image_scale * average_display_dimension) *
                                (uav_image_height / average_rover_dimension))

        self.green_poi_width = int((poi_image_scale*average_display_dimension) *
                                   (green_poi_image_width/average_green_poi_dimension))
        self.green_poi_height = int((poi_image_scale*average_display_dimension) *
                                    (green_poi_image_height/average_green_poi_dimension))

        self.red_poi_width = int((poi_image_scale*average_display_dimension) *
                                 (red_poi_image_width/average_red_poi_dimension))
        self.red_poi_height = int((poi_image_scale*average_display_dimension) *
                                  (red_poi_image_height/average_red_poi_dimension))

        self.circle_radius = int(circle_radius_ratio*average_display_dimension)
        self.line_width = int(line_width_ratio*average_display_dimension)

        self.observation_radius = observation_radius
        #self.ring_diameter = int(4*observation_radius*average_display_dimension/(self.grid_width + self.grid_height)) #todo: need to see this

        # scaling down the images
        self.background_image = pygame.transform.scale(self.background_image, (self.display_width, self.display_height)) # We obtain the scaled background image
        self.rover_image = pygame.transform.scale(self.rover_image, (self.rover_width, self.rover_height)) # We obtain the scaled rover image
        self.uav_image = pygame.transform.scale(self.uav_image, (self.rover_width, self.rover_height)) # We obtain the scaled rover image
        self.green_poi_image = pygame.transform.scale(self.green_poi_image, (self.green_poi_width, self.green_poi_height))         # We obtain the scaled green poi image
        self.red_poi_image = pygame.transform.scale(self.red_poi_image, (self.red_poi_width, self.red_poi_height))
        #self.observation_ring_image = pygame.transform.scale(self.observation_ring_image, (self.ring_diameter, self.ring_diameter))

        # The effective display dimensions used to prevent clipping of rover and POI images placed near the vertical edges# todo: this is wow
        self.effective_display_width = self.display_width - np.max([self.rover_width, self.green_poi_width,
                                                                    self.red_poi_width])
        self.effective_display_height = self.display_height - np.max([self.rover_height, self.green_poi_height,
                                                                      self.red_poi_height])

        # List of RGB colors for marking the path of each rover
        self.trail_color_array = self.generate_random_colors(rover_count + uav_count)
        self.rover_trail_list = []
        self.poi_status_list = []


    def update_visualizer(self, rover_pos_list, poi_pos_list, reset=False, wait_time=0.1):
        '''
        This function updates the pygame display with the passed parameters
        rover_pos_list: the rover coordinates as a list  eg: [(1,2),(2,3),(3,4)]
        poi_pos_list: POI coordinates as a list  eg: [(1,2),(2,3),(3,4)]
        poi_status_list: a list of the observation status of the POIs eg: [True, False, True]
        Reset: used if we want to clear the rover trails when we end the episode
        Wait_time: time_delay to pause between frames
        We draw the background on the screen to erase everything previously drawn over the window
        '''

        self.pygame_display.blit(self.background_image, (0, 0))

        self.pygame_display.blit(self.text_surface, ((self.display_width - self.text_surface_width) / 2,
                                                     self.display_height - 1.5 * self.text_surface_height))

        # Display the observation ring around each rover
        if display_rings:
            for i in range(len(poi_pos_list)):
                poi_coordinate = poi_pos_list[i]
                adjusted_x_coordinate = int(self.effective_display_width * (poi_coordinate[0] / self.grid_width))
                adjusted_y_coordinate = int(self.effective_display_height * (poi_coordinate[1] / self.grid_height))
                self.pygame_display.blit(self.observation_ring_image,
                                         (adjusted_x_coordinate-self.ring_diameter/2+self.green_poi_width/2,
                                          adjusted_y_coordinate-self.ring_diameter/2+self.green_poi_height/2))

        adjusted_rover_pos_list = []
        # Plot the path of each rover
        for coordinate in rover_pos_list:
            adjusted_x_coordinate = int(self.effective_display_width * (coordinate[0] / self.grid_width))
            adjusted_y_coordinate = int(self.effective_display_height * (coordinate[1] / self.grid_height))
            adjusted_rover_pos_list.append((adjusted_x_coordinate, adjusted_y_coordinate))

        # If the reset flag is True, we should reset the rover trail list
        if reset:
            # This list holds lists (for each time step) containing the coordinates of each rover
            self.rover_trail_list = []
            self.rover_trail_list.append(adjusted_rover_pos_list)
        else:
            self.rover_trail_list.append(adjusted_rover_pos_list)

        # Iterating through the time steps recorded
        for time_step in range(len(self.rover_trail_list)):
            # Iterate through all the rovers
            for rov_id in range(len(adjusted_rover_pos_list)):
                rgb_color = self.trail_color_array[rov_id]
                adjusted_rover_pos = self.rover_trail_list[time_step][rov_id]
                if time_step == len(self.rover_trail_list)-1:
                    self.pygame_display.blit(self.rover_image, adjusted_rover_pos)
                else:
                    pygame.draw.circle(self.pygame_display, rgb_color,
                                       (adjusted_rover_pos[0] + int(self.rover_width / 2),
                                        adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                       self.circle_radius)

                    # Obtain the next step
                    next_step_adjusted_rover_pos = self.rover_trail_list[time_step+1][rov_id]
                    # Draw the path of the rover
                    pygame.draw.line(self.pygame_display, rgb_color,
                                     (adjusted_rover_pos[0] + int(self.rover_width / 2),
                                      adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                     (next_step_adjusted_rover_pos[0] + int(self.rover_width / 2),
                                      next_step_adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                     self.line_width)

        # We update the POI statuses
        self.update_poi_status(rover_pos_list, poi_pos_list, reset)

        # We iterate through the poi list
        for i in range(len(poi_pos_list)):
            poi_coordinate = poi_pos_list[i]
            poi_status = self.poi_status_list[i]
            adjusted_x_coordinate = int(self.effective_display_width * (poi_coordinate[0] / self.grid_width))
            adjusted_y_coordinate = int(self.effective_display_height * (poi_coordinate[1] / self.grid_height))
            # If the poi is observed
            if poi_status:
                self.pygame_display.blit(self.green_poi_image, (adjusted_x_coordinate, adjusted_y_coordinate))
            else:
                self.pygame_display.blit(self.red_poi_image, (adjusted_x_coordinate, adjusted_y_coordinate))

        pygame.display.update()
        time.sleep(max(wait_time, 0.0333))

    # This function updates the POI status
    def update_poi_status(self, rover_pos_list, poi_pos_list, reset=False):
        if not self.poi_status_list or reset:
            # We use a POI status list of all False
            self.poi_status_list = [False for _ in range(len(poi_pos_list))]
        for i in range(len(poi_pos_list)):
            if self.poi_status_list[i]:
                continue
            poi_coordinate = poi_pos_list[i]
            coupled_rovers = 0
            coupled_uavs = 0

            for rover_id, coordinate in enumerate(rover_pos_list):
                euclidean = math.sqrt((poi_coordinate[0]-coordinate[0])**2 + (poi_coordinate[1]-coordinate[1])**2)
                if euclidean <= self.observation_radius:
                    if(rover_id < self.uav_count): # first UAvs
                        coupled_uavs += 1
                    else:
                        coupled_rovers += 1 #for rovers

                if coupled_rovers == self.coupling_factor_rover and coupled_uavs == self.coupling_factor_uav:
                    self.poi_status_list[i] = True
                    break

    @staticmethod
    # This function generates the specified number of random colors as an list
    def generate_random_colors(number_of_colors, default=((255, 0, 0), (0, 200, 0), (0, 0, 255), (128, 0, 128))):
        # The list of colors is initialized to an empty list
        color_list = []
        for i in range(number_of_colors):
            if i <= len(default):
                # We append the default colors
                color_list.append(default[i])
            else:
                # We append a random RGB vector eg:[1, 200, 135] to the color list
                color_list.append(list(np.random.choice(range(256), size=3)))
        return color_list

# -------------------------------- CODE FOR TESTING ONLY --------------------------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # Parameters for testing
    rover_count = 1
    uav_count = 1
    poi_count = 1
    rounds_to_run = 1000
    reset_interval = 50
    step_size = 1
    grid_size = (20, 20)
    observation_radius = 5
    coupling_factor_rover = 1
    coupling_factor_uav = 1
    poi_prob = 0.05
    waiting_time = 0.1

    # We initialize the visualization object with the number of rovers and the grid size (width x height)
    visualizer = RoverDomainVisualizer(rover_count, uav_count, grid_size, observation_radius, coupling_factor_uav, coupling_factor_rover)

    rover_coordinate_list = []
    poi_coordinate_list = []
    poi_status_list = []

    # We iterate through the number of rounds to run
    for i in range(rounds_to_run):
        # We check if it is the round to remove the trails
        if i % reset_interval == 0:
            reset = True
            rover_coordinate_list = [(grid_size[0] * random.random(), grid_size[1] * random.random()) for _ in range(rover_count)]
            poi_coordinate_list = [(grid_size[0] * random.random(), grid_size[1] * random.random()) for _ in range(poi_count)]
        else:
            reset = False

        # We update the visualizer
        visualizer.update_visualizer(rover_coordinate_list, poi_coordinate_list, reset, waiting_time)

        # Here we move the rovers for the next round
        for j in range(rover_count):
            rover_x = rover_coordinate_list[j][0]
            rover_y = rover_coordinate_list[j][1]
            rover_x += step_size*random.random() - step_size/2
            rover_y += step_size*random.random() - step_size/2
            rover_x = min(grid_size[0], max(0, rover_x))
            rover_y = min(grid_size[1], max(0, rover_y))
            rover_coordinate_list[j] = (rover_x, rover_y)
