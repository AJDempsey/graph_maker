# Unified SAOS Topology Creator Functional Spec

## Revision history

| version | Comment |
|---------|---------|
| 0.1     | Initial commit |

## Introduction

This document will define the functional and non-functional requirements for a unified  
SAOS topology creator app called saos_topo. This app will be a TUI - a command line tool  
with a user interface - that the user can control using either a mouse or a keyboard.  
This document serves as the primary reference for me (Anthony Dempsey) and others in  
the IPServices UK team.

The specification will cover - the functionality of the user interface; the methods used  
to control the various specifications of SAOS nodes; initial configuration of nodes;  
updating running code on a topology; how the app tracks user topologies.

* In scope
  * UI flow from initial start up, through designer workflow, and closing the app
  * User created topology management
    * Listing created topologies
    * Starting a new topology with a name
    * Stopping a named topology
    * Probing a named topology for IP addresses
    * Deploying new code to one or more nodes in the topology
  * Creation of a new named topology from a library of definitions
  * Adding config templates to be used on nodes
  * New topology definition creation
* Out of scope
  * Additional configuration application after initial configuration
  * Attaching to hardware that's not in the format that the Edinburgh IPService uses
  * Hardware reservation control
  * Running tests

### Glossary

| Abberviation | Expansion |
|--------------|------------|
| SAOS | Service Aware Operating System |
| TUI | Terminal User Interface |

## Overview

There's three separate ways that a SAOS node can exists, OneContainer, Simulator, and actual  
hardware. Of these three methods OneC has slightly different methods of configuration but  
all of them can all be configured in the same manner. Rather than having separate workflows  
for each of them there should be a unified method to:

* define a topology
* add configuration
* control the lifecycle of the machines
* change the running code on any number of nodes

Given that all of the methods of creating a SAOS like node happen on the command line  
there's an opportunity to create a TUI as a unified manager for these workflows.  

