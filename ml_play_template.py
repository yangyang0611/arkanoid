"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

import pickle
import math
import numpy as np
import os.path

def ml_loop():
	"""The main loop of the machine learning process

	This loop is run in a seperate process, and communicates with the game process.

	Note that the game process won't wait for the ml process to generate the
	GameInstrcution. It is possible that the frame of the GameInstruction
	is behind of the current frame in the game process. Try to decrease the fps
	to avoid this situation.
	"""

	# === Here is the execution order of the loop === #
	# 1. Put the initialization code here.

	# 2. Inform the game process that ml process is ready before start the loop.


	filename = "newrule_model.sav"
	filepath = os.path.join(os.path.dirname(__file__), filename)
	load_model = pickle.load(open(filepath, 'rb'))
	comm.ml_ready()
	Vx = 7
	Vy = 7
	predict_platX = 75
	# map_empty = np.zeros((40, 9), dtype=np.int)
	# 3. Start an endless loop.
	while True:
		# 3.1. Receive the scene information sent from the game process.
		scene_info = comm.get_scene_info()

		# 3.2. If the game is over or passed, the game process will reset
		#      the scene immediately and send the scene information again.
		#      Therefore, receive the reset scene information.
		#      You can do proper actions, when the game is over or passed.
		if scene_info.status == SceneInfo.STATUS_GAME_OVER or \
			scene_info.status == SceneInfo.STATUS_GAME_PASS:
			scene_info = comm.get_scene_info()		

		# 3.3. Put the code here to handle the scene information
		if scene_info.frame == 0:
			y_direction = 0
			y_last_position = scene_info.ball[1]
			x_direction = 0
			x_last_position = scene_info.ball[0]
			x_last_direction = 0
			x_rebound = 0
			y_rebound = 0

		else:
			distance = lambda x: 1 if x > 0 else -1
			y_direction = distance(scene_info.ball[1] - y_last_position)
			x_direction = distance(scene_info.ball[0] - x_last_position)
			if scene_info.ball[0] - x_last_position != Vx:
				x_rebound = 1
				Vx = -Vx
			else:
				x_rebound = 0
			if scene_info.ball[1] - y_last_position != Vy:
				y_rebound = 1
				Vy = -Vy
			else:
				y_rebound = 0
		if y_direction == -1:
			x_rebound = 0
		if scene_info.ball[1] < 200:
			y_rebound = 0
		# if x_rebound !=0 and (scene_info.ball[0] >= 10 and scene_info.ball[0] <= 185):
		# 	x_rebound = 0
		# print(scene_info.ball, x_rebound, Vx)
		# My rule for predict position	
		if y_rebound != 0:
			x_temp = scene_info.ball[0]
			y_temp = scene_info.ball[1]
			times = (int)(math.ceil((395 - y_temp) / abs(Vx)))
			predict_ballx = x_temp + times * Vx
			if predict_ballx < 0:
				times = (int)(math.floor((0 - predict_ballx) / abs(Vx)))
				predict_ballx = abs(Vx) * times
			if predict_ballx > 195:
				times = (int)(math.floor((predict_ballx - 195) / abs(Vx)))
				predict_ballx = 195 - abs(Vx) * times
			predict_platX = predict_ballx - 17
			# print('Predict ball: ', predict_ballx)
			# print(y_direction)


		if x_rebound != 0:
			x_temp = scene_info.ball[0]
			y_temp = scene_info.ball[1]
			times = (int)(math.ceil((395 - y_temp) / abs(Vx)))
			predict_ballx = x_temp + times * Vx
			if predict_ballx < 0:
				times = (int)(math.floor((0 - predict_ballx) / abs(Vx)))
				predict_ballx = abs(Vx) * times
			if predict_ballx > 195:
				times = (int)(math.floor((predict_ballx - 195) / abs(Vx)))
				predict_ballx = 195 - abs(Vx) * times
			# print(predict_platX)
			predict_platX = predict_ballx - 17
		# print([scene_info.ball[0], scene_info.ball[1]])
		# print(predict_platX)
		# print(predict_platX)
		# print(input_current)
		# 3.4. Send the instruction for this frame to the game process
		if scene_info.ball[1] < 200 or y_direction == -1:
			if scene_info.platform[0] > 75:
				instruct = -1
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)
			elif scene_info.platform[0] < 75:
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_RIGHT)
				instruct = 1
			else:
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_NONE)
				instruct = 0
		else:
			if scene_info.platform[0] > predict_platX:
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)
				instruct = -1
			elif scene_info.platform[0] < predict_platX:
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_RIGHT)
				instruct = 1
			else:
				# comm.send_instruction(scene_info.frame, GameInstruction.CMD_NONE)
				instruct = 0
		input_current = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0], y_direction, instruct])
		inp_current = input_current[np.newaxis, :]
		if load_model.predict(inp_current) == 1:
			comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
		elif load_model.predict(inp_current) == -1:
			comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
		else:
			comm.send_instruction(scene_info.frame, PlatformAction.NONE)


		x_last_position = scene_info.ball[0]
		y_last_position = scene_info.ball[1]
		x_last_direction = x_direction
		