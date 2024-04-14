#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq as hq
import itertools
import math
import copy

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario

	def find_closest_city(self, city, list_of_cities: list):
		min_dist = math.inf
		min_city = None
		for city in list_of_cities:
			dist = city.costTo(city)
			if dist < min_dist:
				min_city = city
				min_dist = dist
		return min_city

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	# For each starting city, the function tries to find the shortest path that visits all
	# remaining cities using the greedy approach.
	def greedy( self,time_allowance=60.0 ):
		cities = self._scenario.getCities()
		start_time = time.time()
		best_solution = None
		num_solutions = 0

		# Loop through each city as a starting point
		for start_city in cities:
			if time.time() - start_time > time_allowance:
				break

			route = [start_city]
			remaining_cities = set(cities)
			remaining_cities.remove(start_city)

			# Add nearest city to the current city until all cities have been visited
			while remaining_cities:
				next_city = min(remaining_cities, key=lambda city: start_city.costTo(city))
				if start_city.costTo(next_city) == math.inf:
					break

				route.append(next_city)
				remaining_cities.remove(next_city)
				start_city = next_city

			# If valid solution is found, update variables for best solution and number of solutions
			if len(route) == len(cities) and start_city.costTo(route[0]) != math.inf:
				solution = TSPSolution(route)
				num_solutions += 1

				if not best_solution or solution.cost < best_solution.cost:
					best_solution = solution

		end_time = time.time()
		results = {
			'cost': best_solution.cost if best_solution else math.inf,
			'time': end_time - start_time,
			'count': num_solutions,
			'soln': best_solution,
			'max': None,
			'total': None,
			'pruned': None,
		}
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		count, total, maximum, pruned = 0, 0, 0, 0
		greedy_result = self.greedy()
		time_start = time.time()

		state_heap = []

		# get greedy solution and set it as the current best solution
		greedy = greedy_result['soln']
		best_so_far = TSPSolution(greedy.getRoute())
		route_length = len(greedy.getRoute())
		start_state = StateBranchAndBound(greedy.getRoute())
		hq.heappush(state_heap, start_state)
		total += 1

		# loop until there are no more states to explore or time limit is reached
		while len(state_heap) > 0:
			if time.time() - time_start > time_allowance:
				break
			maximum = max(maximum, len(state_heap))

			state = hq.heappop(state_heap)
			if len(state.currentPath) == route_length:
				solution = TSPSolution(state.currentPath)
				if best_so_far.cost > solution.cost:
					best_so_far = solution
					count += 1
				else:
					pruned += 1

			elif (state.lowerBound < math.inf) and (state.lowerBound < best_so_far.cost):
				count += 1
				new_states = state.possible_state()
				for newState in new_states:
					hq.heappush(state_heap, newState)
					total += 1
			else:
				pruned += 1

		end_time = time.time()
		results.update({
			'cost': best_so_far.cost,
			'time': greedy_result['time'] + end_time - time_start,
			'count': count,
			'soln': best_so_far,
			'max': maximum,
			'total': total,
			'pruned': pruned
		})
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		
class StateBranchAndBound:
	def __init__(self, cities_list, lower_bound=0, current_path=None, prev_state=None, test=None):
		if current_path is None:
			current_path = []
		self.listCities = cities_list
		self.lowerBound = lower_bound
		self.currentPath = copy.copy(current_path)
		self.prevState = prev_state
		self.testPath = test
		self.matrix = Table(len(cities_list))
		self.currentPath.append(cities_list[test[1]] if test else cities_list[0])
		for row in range(len(cities_list)):
			for col in range(len(cities_list)):
				if test and (row == test[1] or col == test[0]):
					self.matrix.set_value(row, col, 'x')
				elif test and row == test[0] and col == test[1]:
					self.matrix.set_value(row, col, np.inf)
				elif not test or col != test[0] or row != test[1]:
					matrix_value = cities_list[col].costTo(cities_list[row]) if not prev_state else prev_state.get_value(
						row,
						col)
					self.matrix.set_value(row, col, matrix_value)
		self.lowerBound += self.matrix_reduce()

	def possible_state(self):
		current_path = self.currentPath
		test = self.testPath

		city_list = self.listCities
		state_matrix = self.matrix
		state_list = []

		lower_bound = self.lowerBound
		current_index = 0

		if test is not None:
			current_index = test[1]

		# Looks a tad bit weird but performance gain from doing it this way is very noticeable (0.02-0.1 second speed up)
		state_list += [
			StateBranchAndBound(city_list, lower_bound + state_matrix.get_value(nextIndex, current_index), current_path,
								state_matrix, [current_index, nextIndex])
			for nextIndex in range(len(city_list))
			if city_list[nextIndex] not in current_path and nextIndex != current_index
		]
		return state_list

	def __lt__(self, other):
		return self.lowerBound < other.lowerBound

	def matrix_reduce(self):
		size_matrix = self.matrix.get_size()
		state_matrix = self.matrix
		lower_bound = 0

		# Reduce rows
		for y in range(size_matrix[1]):
			row_min = np.inf
			for x in range(size_matrix[0]):
				test_value = state_matrix.get_value(x, y)
				if test_value == 'x':
					continue
				elif test_value < row_min:
					row_min = test_value
			if row_min != np.inf:
				lower_bound += row_min
				for x in range(size_matrix[0]):
					test_value = state_matrix.get_value(x, y)
					if test_value != 'x':
						state_matrix.set_value(x, y, test_value - row_min)

		# Reduce columns
		for x in range(size_matrix[0]):
			col_min = np.inf
			for y in range(size_matrix[1]):
				test_value = state_matrix.get_value(x, y)
				if test_value == 'x':
					continue
				elif test_value < col_min:
					col_min = test_value

			if col_min != np.inf:
				lower_bound += col_min
				for y in range(size_matrix[1]):
					test_value = state_matrix.get_value(x, y)
					if test_value != 'x':
						state_matrix.set_value(x, y, test_value - col_min)

		return lower_bound