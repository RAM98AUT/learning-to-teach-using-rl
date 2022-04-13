#Plotting Curriculum
import matplotlib.pyplot as plt
import numpy as np


def plot_curriculum(curr,title,n_skills,n_exercises,blocks,max_skills,
                    color1="navajowhite",color2= "papayawhip",ticks=True,size=5):
    """
    Plots a curriculum that shows the actions of an agent

    Parameters
    ----------
    curr : TYPE
        DESCRIPTION.
    title : str
        Title of the output picture
    n_skills : int
        number of skills (y-Axis)
    n_exercises : int
        number of exercises (x-Axis)
    blocks : list
        blockstructure
    max_skills : int
        maximum of skills per exercise
    color1 : str, optional
        First block colour. The default is "navajowhite".
    color2 : str, optional
        Second block colour. The default is "papayawhip".
    ticks: boolean, optional
        Defines if ticks on y axis should be printed. The default is True

    Returns
    -------
    None.

    """
    beg = -0.5
    end = -0.5
    for num,x in enumerate(blocks):
        end += x
        plt.axhspan(beg,end, facecolor=_colorblocks(num,color1,color2))
        beg = end
    for i in range(max_skills):
        plt.plot(range(n_exercises),curr[:,i],"bo",markersize=size)
    
    if ticks:
        plt.yticks(np.arange(0, n_skills, 1))  
    plt.ylim([-0.5, n_skills-0.5])
    plt.xlabel("Exercises")
    plt.ylabel("Skills")
    plt.title(title)
    plt.show()
    #data_str='output/'+title+".png"
    #plt.savefig(data_str)

def _colorblocks(number,color1,color2):
    """
    Returns if block should have colour 1 or 2

    Parameters
    ----------
    number : int
        Number of the respective block
    color1 : str, optional
        First block colour. 
    color2 : str, optional
        Second block colour. 

    Returns
    -------
    str
        Colour 1 or 2

    """
    if number%2 == 0:
        return color1
    else:
        return color2