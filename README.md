<!-- PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">Unreal Engine Deep Reinforcement Learning Library</h3>

  <p align="center">
    Library and plugin for streamlined development of learning agents in Unreal Engine 5
   <br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#getting-started">Getting Started</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Reinforcement learning framework designed specifically for integration with Unreal Engine.

Made for both developers and non-programmers, the library supports native code alongside Blueprint visual scripting, enabling users to implement learning agents within their Unreal projects regardless of their programming background.

The framework features a minimal setup process to get you started fast. Its design makes it straightforward to set up training environments and experiment with different reinforcement learning algorithms without hassle.



### Built With

![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-0E1128?logo=unrealengine&logoColor=fff&style=for-the-badge)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=fff&style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

<!-- INSTALLATION -->

## Installation

Follow the instructions below to get started.

### Prerequisites

[Unreal Engine 5.3 and above](https://www.unrealengine.com/en-US/download)


### Getting Started

Open your terminal and clone the repository to your local machine:

`git clone https://github.com/kcccr123/ue-reinforcement-learning.git`

The repository is split into an external python module and an Unreal Editor project plugin. 

#### Unreal Setup

1. Copy the entire plugin folder, `UnrealPlugin` into the Plugins directory of your Unreal project.

If the Plugins folder doesn’t exist, create one in the root of your project. The folder structure of your Unreal project should look like this: 

```plaintext
YourUnrealProject/
└── Plugins/
    └── UnrealPlugin/
```
2. Enable the plugin inside the Unreal Editor.
   
Open your Unreal project inside the editor. Navigate to Edit > Plugins. Locate your plugin in the list (it might be under a relevant category such as "Other" or "Installed Plugins"). Check the box to enable the plugin. Restart Unreal Engine if prompted.

#### Python Setup

It is recommended you create a virtual enviornment to keep track of your packages. 

## Usage


<!-- CONTACT -->

## Contact

Feel free to contact us at:

@Kevin Chen - kevinz.chen@mail.utoronto.ca\

<p align="right">(<a href="#readme-top">back to top</a>)</p>
