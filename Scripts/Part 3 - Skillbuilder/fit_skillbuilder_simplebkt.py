"""
Fits a separate simplebkt model for every skill in the skillbuilder dataset.
"""
from torchbkt import *
from datahelper import *
from datahelper.constants import skillbuilder_n_skills, skillbuilder_max_skills


# here correct path has to be added     
path = "data/skill_builder_data_corrected_collapsed.csv"


optimization_params = {
    'lr': 0.01,
    'epochs': 10,
    'batch_size': 8
}


if __name__=='__main__':
    importer = SkillbuilderImporter()
    skillbuilder = importer(path)
    skillbuilder = remove_null(skillbuilder)
    skillbuilder = remove_rare_skills(skillbuilder)
    skill_mapping = get_skill_mapping(skillbuilder, skillbuilder_max_skills)
    skillbuilder = apply_skill_mapping(skillbuilder, skillbuilder_max_skills, skill_mapping)
    model = BKTFitMultiple(skillbuilder, skillbuilder_n_skills, skillbuilder_max_skills, **optimization_params)
    df = model.fitting_multiple(verbose=True)