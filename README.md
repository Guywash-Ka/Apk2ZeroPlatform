## Android Code Extraction Project
This project is a comprehensive tool for extracting platform-independent code from Android applications. It analyzes Java source code to identify Android dependencies and uses LLM to transform Android-dependent code into platform-independent Java.

### Files Overview
The project consists of several Python modules that work together:

__android_dependencies.py__ - Defines Android package and component patterns for dependency detection

__calculate_kpi.py__ - Script to calculate key performance indicators for code transformation

__java_analyzer.py__ - Analyzes Java files to identify Android dependencies

__kpi.py__ - Contains metrics and evaluation functions for measuring transformation quality

__llm_transformer.py__ - Uses LLM to transform Android-dependent code to platform-independent Java

### Setup
#### Requirements
Python 3.7+

#### Environment Variables
The project requires API keys for LLM services. Set them as environment variables:

`export LLM_API_KEY=your_api_key_here`

### Usage
#### Analyzing Java Files

`python java_analyzer.py --directory path/to/java/files --output analysis_output`

#### Transforming Android-Dependent Code

`python llm_transformer.py --api-key $LLM_API_KEY --model gpt-4o`

#### Calculating KPIs

`python calculate_kpi.py`

### License
This project is for educational and research purposes.
