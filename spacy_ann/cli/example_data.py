# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import srsly
import typer
from wasabi import Printer


def example_data(output_dir: Path, verbose: bool = False):
    """Download Example Data from Github

    output_dir (Path): path to output_dir for entities.jsonl and aliases.jsonl
    """
    msg = Printer(hide_animation = not verbose)

    msg.divider("Example Data")
    with msg.loading(f"Writing Example data to {output_dir}"):

        aliases_data = [
            {"alias": "ML", "entities": ["a1", "a2"], "probabilities": [0.5, 0.5]},
            {"alias": "Machine learning", "entities": ["a1"], "probabilities": [1.0]},
            {"alias": "Meta Language", "entities": ["a2"], "probabilities": [1.0]},
            {"alias": "NLP", "entities": ["a3", "a4"], "probabilities": [0.5, 0.5]},
            {"alias": "Natural language processing", "entities": ["a3"], "probabilities": [1.0]},
            {"alias": "Neuro-linguistic programming", "entities": ["a4"], "probabilities": [1.0]},
            {"alias": "Operating system", "entities": ["a5"], "probabilities": [1.0]},
            {"alias": "OS", "entities": ["a5"], "probabilities": [1.0]},
            {"alias": "Statistics", "entities": ["a6"], "probabilities": [1.0]},
            {"alias": "Audience segmentation", "entities": ["a7"], "probabilities": [1.0]},
            {"alias": "Decision analysis", "entities": ["a8"], "probabilities": [1.0]},
            {"alias": "Computer science", "entities": ["a9"], "probabilities": [1.0]},
            {"alias": "Photochemistry", "entities": ["a10"], "probabilities": [1.0]},
            {"alias": "Mineralogy", "entities": ["a11"], "probabilities": [1.0]},
            {"alias": "Stereochemistry", "entities": ["a12"], "probabilities": [1.0]},
            {"alias": "Environmental chemistry", "entities": ["a13"], "probabilities": [1.0]},
            {"alias": "Agronomy", "entities": ["a14"], "probabilities": [1.0]},
            {"alias": "Research", "entities": ["a15"], "probabilities": [1.0]}
        ]

        entities_data = [
            {"id":"a1","name":"Machine learning (ML)","description":"Machine learning (ML) is the scientific study of algorithms and statistical models..."},
            {"id":"a2","name":"ML (\"Meta Language\")","description":"ML (\"Meta Language\") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as \"Lisp with types\"."},
            {"id":"a3","name":"Natural language processing (NLP)","description":"Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."},
            {"id":"a4","name":"Neuro-linguistic programming (NLP)","description":"Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States in the 1970s."},
            {"id":"a5","name":"Operating system","description":"Operating Systems consists of building system software that provides common services for other types of computer programs.","label":"SKILL"},
            {"id":"a6","name":"Statistics","description":"Statistics deals with all aspects of data collection, organization, analysis, interpretation, and presentation.","label":"SKILL"},
            {"id":"a7","name":"Audience segmentation","description":"Audience segmentation is a process of dividing people into homogeneous subgroups based upon defined criterion such as product usage, demographics, psychographics, communication behaviors and media use. Audience segmentation is used in commercial marketing so advertisers can design and tailor products and services that satisfy the targeted groups. In social marketing, audiences are segmented into subgroups and assumed to have similar interests, needs and behavioral patterns and this assumption allows social marketers to design relevant health or social messages that influence the people to adopt recommended behaviors. Audience segmentation is widely accepted as a fundamental strategy in communication campaigns to influence health and social change. Audience segmentation makes campaign efforts more effective when messages are tailored to the distinct subgroups and more efficient when the target audience is selected based on their susceptibility and receptivity.","label":"SKILL"},
            {"id":"a8","name":"Decision analysis","description":"Decision analysis (DA) is the discipline comprising the philosophy, methodology, and professional practice necessary to address important decisions in a formal manner. Decision analysis includes many procedures, methods, and tools for identifying, clearly representing, and formally assessing important aspects of a decision, for prescribing a recommended course of action by applying the maximum expected utility action axiom to a well-formed representation of the decision, and for translating the formal representation of a decision and its corresponding recommendation into insight for the decision maker and other stakeholders.","label":"SKILL"},
            {"id":"a9","name":"Computer science","description":"Computer science is the study of processes that interact with data and that can be represented as data in the form of programs. It enables the use of algorithms to manipulate, store, and communicate digital information. A computer scientist studies the theory of computation and the practice of designing software systems."},
            {"id":"a10","name":"Photochemistry","description":"Photochemistry is the branch of chemistry concerned with the chemical effects of light. Generally, this term is used to describe a chemical reaction caused by absorption of ultraviolet (wavelength from 100 to 400 nm), visible light (400\u2013750 nm) or infrared radiation (750\u20132500 nm).","label":"SKILL"},
            {"id":"a11","name":"Mineralogy","description":"Mineralogy is a subject of geology specializing in the scientific study of the chemistry, crystal structure, and physical (including optical) properties of minerals and mineralized artifacts. Specific studies within mineralogy include the processes of mineral origin and formation, classification of minerals, their geographical distribution, as well as their utilization.","label":"SKILL"},
            {"id":"a12","name":"Stereochemistry","description":"Stereochemistry, a subdiscipline of chemistry, involves the study of the relative spatial arrangement of atoms that form the structure of molecules and their manipulation. The study of stereochemistry focuses on stereoisomers, which by definition have the same molecular formula and sequence of bonded atoms (constitution), but differ in the three-dimensional orientations of their atoms in space. For this reason, it is also known as 3D chemistry\u2014the prefix \"stereo-\" means \"three-dimensionality\".","label":"SKILL"},
            {"id":"a13","name":"Environmental chemistry","description":"Environmental chemistry is the scientific study of the chemical and biochemical phenomena that occur in natural places. It should not be confused with green chemistry, which seeks to reduce potential pollution at its source. It can be defined as the study of the sources, reactions, transport, effects, and fates of chemical species in the air, soil, and water environments; and the effect of human activity and biological activity on these. Environmental chemistry is an interdisciplinary science that includes atmospheric, aquatic and soil chemistry, as well as heavily relying on analytical chemistry and being related to environmental and other areas of science.","label":"SKILL"},
            {"id":"a14","name":"Agronomy","description":"Agronomy is the science and technology of producing and using plants for food, fuel, fiber, and land restoration. Agronomy has come to encompass work in the areas of plant genetics, plant physiology, meteorology, and soil science. It is the application of a combination of sciences like biology, chemistry, economics, ecology, earth science, and genetics. Agronomists of today are involved with many issues, including producing food, creating healthier food, managing the environmental impact of agriculture, and extracting energy from plants. Agronomists often specialise in areas such as crop rotation, irrigation and drainage, plant breeding, plant physiology, soil classification, soil fertility, weed control, and insect and pest control.","label":"SKILL"},
            {"id":"a15","name":"Research","description":"Research is \"creative and systematic work undertaken to increase the stock of knowledge, including knowledge of humans, culture and society, and the use of this stock of knowledge to devise new applications.\" or in other hand Research is a process of steps used to collect and analyze information to increase our understanding of a topic or issue. At a general level, research consists of three steps: 1. Pose a question. 2. Collect data to answer the question. 3. Present an answer to the question. This should be a familiar process. You engage in solving problems every day and you start with a question, collect some information, and then form an answer\nResearch is important for three reasons.1. Research adds to our knowledge: Adding to knowledge means that educators undertake research to contribute to existing information about issues 2.Research improves practice: Research is also important because it suggests improvements for practice. Armed with research results, teachers and other educators become more effective professionals. 3. Research informs policy debates: research also provides information to policy makers when they research and debate educational topics.","label":"SKILL"}
        ]

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        srsly.write_jsonl(output_dir / "entities.jsonl", entities_data)
        srsly.write_jsonl(output_dir / "aliases.jsonl", aliases_data)
        msg.good("Done.")
    

if __name__ == "__main__":
    typer.run(example_data)
