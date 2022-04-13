# Learning to Teach Using Reinforcement Learning
In this project, we deal with the topic of personalized education, i.e. ensuring that students acquire knowledge about some subject(s) with an individualized sequence of exercises for learning. The project contains two major parts: One part is all about fitting knowledge tracing models that take into account how hard certain skills are to learn and to apply, and can maintain an estimate of the current knowledge of a student during an exercise sequence. The other part is about finding reinforcement learning agents that learn to select exercises to make the knowledge acquisition process as efficient as possible. Last but not least, we deal with the fairness aspect that arises when students with different speed of learning are to be taught. Please have a look at the report in the `documentation` folder for more information on the topic.

## Prerequisites
With the requirements listed in the `environment.yaml` file, you should be able to run all the code included.
If you wish, you can create a separate conda environment with the command `conda env create -f environment.yaml`.

## File Structure
The folders `datahelper`, `figurelib`, `RL`, `simulator` and `torchbkt` contain Python packages with our actual implementation.
The `data` and the `output` folders contain our input and output data.
You have to create the `data` folder yourself and ideally place the `skill_builder_data_corrected_collapsed.csv` from the Skillbuilder download website in this folder. Further data can be generated with the `produce_figure.py` file in `Scripts/Part 1 - Figure Fairness` by setting the variable `save_data` to `True` at the beginning of the script. In the `Scripts` folder, you can find various scripts to try out our implementation and produce results. All scripts are supposed to have a docstring at the top explaining what the respective script can be used for. The folders `notebooks` and `notes` contain some exploration and documentation, particularly our final project report. 

## References
* https://colab.research.google.com/drive/1IUe9lfoIiQsL49atSOgxnCmMR_zJazKI
* https://www.youtube.com/watch?v=8ua0qfbPnfk
* see `report.pdf` for further references ...