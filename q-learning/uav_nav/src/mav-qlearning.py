#!/usr/bin/env python
import gym
import rospy
import mav_env
from gym import wrappers
import gym_gazebo
import time
import random
import time
import liveplot
import qlearn

def save_frames_as_gif(frames, path='/home/ese650_ws/src/uav_nav/outdir/qlearning/', filename='gym_animation.gif'):

    #Mess with this to change frame size
    print("Saving gifs....")
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

if __name__ == '__main__':

    rospy.init_node('qlearning', anonymous=True)
    env = gym.make('MavEnv-v0')
    print "Gym Make Done"


    outdir = '/home/ese650_ws/src/uav_nav/outdir/qlearning'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    print "Monitor Wrapper Started"

    plotter = liveplot.LivePlot(outdir)

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=0.8, gamma=0.9, epsilon=0.9)

    epsilon_discount = 0.9
	
    start_time = time.time()
    total_episodes = 1000
    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 

        print("Episode = " + str(x) + " started")

        observation, done = env.reset()

        if qlearn.epsilon > 0.01:
            qlearn.epsilon *= epsilon_discount

        state = ''.join(map(str, observation))

        i = 0
        while(True):
            action = qlearn.chooseAction(state)
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            nextState = ''.join(map(str, observation))
            qlearn.learn(state, action, reward, nextState)
            env._flush(force=True)
            if not(done):
                state = nextState
            else:
                break
            i = i+1

	    plotter.plot(env)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+"| Reward: "+str(cumulated_reward)+" | Steps: "+str(i+1)+"  | Time: %d:%02d:%02d" % (h, m, s))
	
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(qlearn.epsilon)+"")
    print ("Q table: ")
    print(qlearn.q)
    env.close()
    # save_frames_as_gif(frames)
    plotter.show()
