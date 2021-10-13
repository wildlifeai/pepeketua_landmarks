==================
Landmark detection - Pepeketua ID
==================

Pepeketua ID is an open-source pattern recognition software that facilitates the individual identification of New Zealand endemic frogs (Pepeketua).

.. image:: images/Pepeketua_id_overview.png
   :align: center
   :alt: "(Overview of the three main modules and the components of the Koster Seafloor Observatory.")
    
You can find out more about the project at https://www.wildlife.ai/projects/pepeketua-id/

Overview
------------

This repository contains scripts related to the Landmark Detection component of the Pepeketua ID. 

The Landmark Detection model predicts six frog landmarks in an image. These landmarks will be used to morph and crop the image to facilitate the individual identification of each frog.

.. image:: images/landmark_example_labelled.jpg
   :align: center
   :width: 400
   :alt: "(Example of the frog landmarks predicted by the model.")
   
Example of the landmarks (red points) produced by the model.  

Quickstart
--------------------

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/wildlifeai/pepeketua_landmarks/HEAD


Requirements
------------

* Python 3.7+
* Python dependencies listed in requirements.txt

Instructions
-------------------------
Download the repo, and after building the  docker with build_docker.sh, run run_docker.sh <dir_path> where dir path is a path to your dir with all the images.
It will automaticly find all images in sub dirs as well.

Citation
--------

If you use this code or its models in your research, please cite:

Hay G, Carmon E, Vinograd B, Anton V (2021). An open-source landmark detection model approach to facilitate the individual identification of New Zealand endemic frogs. https://github.com/wildlifeai/pepeketua_landmarks


Collaborations/questions
~~~~~~~~~~~~

We are working to make our work available to other herpetologits. Please feel free to `contact us`_ with your questions.

.. _contact us: contact@wildlife.ai
