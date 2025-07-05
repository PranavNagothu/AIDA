import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Any, Optional, TypedDict
import json
from datetime import datetime
import logging

# LangGraph and LLM imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisState(TypedDict):
    """State structure for the analysis workflow"""
    dataset: pd.DataFrame
    dataset_info: Dict[str, Any]
    column_analysis: Dict[str, Any]
    insights: List[str]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    current_step: str
    error_messages: List[str]

class DataAnalysisAgent:
    def __init__(self, groq_api_key: str, model_name: str = "llama3-70b-8192"):
        """Initialize the Data Analysis Agent"""
        # Fixed: Use correct model name format
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,  # Fixed: Use standard model names
            temperature=0.1,
            max_tokens=2000
        )
        
        # Set up the analysis workflow graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for data analysis"""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes for each analysis step
        workflow.add_node("data_profiler", self._profile_dataset)
        workflow.add_node("column_analyzer", self._analyze_columns)
        workflow.add_node("insight_generator", self._generate_insights)
        workflow.add_node("visualization_planner", self._plan_visualizations)
        workflow.add_node("chart_creator", self._create_charts)
        workflow.add_node("recommendation_engine", self._generate_recommendations)
        
        # Define the workflow edges
        workflow.add_edge("data_profiler", "column_analyzer")
        workflow.add_edge("column_analyzer", "insight_generator")
        workflow.add_edge("insight_generator", "visualization_planner")
        workflow.add_edge("visualization_planner", "chart_creator")
        workflow.add_edge("chart_creator", "recommendation_engine")
        workflow.add_edge("recommendation_engine", END)
        
        # Set entry point
        workflow.set_entry_point("data_profiler")
        
        return workflow.compile()
    
    def _profile_dataset(self, state: AnalysisState) -> AnalysisState:
        """Profile the dataset to understand its structure and characteristics"""
        logger.info("Profiling dataset...")
        
        try:
            df = state["dataset"]
            
            # Basic dataset information
            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},  # Fixed: Convert to string
                "memory_usage": int(df.memory_usage(deep=True).sum()),  # Fixed: Convert to int
                "null_counts": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),  # Fixed: Convert to int
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Use LLM to generate initial insights about the dataset
            prompt = f"""
            Analyze this dataset profile and provide initial observations:
            
            Dataset Shape: {dataset_info['shape']}
            Columns: {dataset_info['columns']}
            Data Types: {dataset_info['dtypes']}
            Missing Values: {dataset_info['null_counts']}
            Duplicate Rows: {dataset_info['duplicate_rows']}
            
            Provide a brief analysis of the dataset structure, data quality issues, and potential analysis opportunities.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            dataset_info["llm_profile"] = response.content
            
            state["dataset_info"] = dataset_info
            state["current_step"] = "data_profiler"
            
        except Exception as e:
            logger.error(f"Error in data profiling: {str(e)}")
            # Ensure error_messages exists and add fallback dataset_info
            if "error_messages" not in state:
                state["error_messages"] = []
            if "dataset_info" not in state:
                state["dataset_info"] = {}
            
            # Add basic fallback info
            try:
                df = state["dataset"]
                state["dataset_info"] = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "memory_usage": int(df.memory_usage(deep=True).sum())
                }
            except Exception:
                # Ultimate fallback
                state["dataset_info"] = {
                    "shape": [0, 0],
                    "columns": [],
                    "dtypes": {},
                    "numeric_columns": [],
                    "categorical_columns": [],
                    "datetime_columns": [],
                    "null_counts": {},
                    "duplicate_rows": 0,
                    "memory_usage": 0
                }
            
            state["error_messages"].append(f"Data profiling error: {str(e)}")
            
        return state
    
    def _analyze_columns(self, state: AnalysisState) -> AnalysisState:
        """Analyze individual columns in detail"""
        logger.info("Analyzing columns...")
        
        try:
            df = state["dataset"]
            column_analysis = {}
            
            for column in df.columns:
                col_data = df[column]
                
                analysis = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isnull().sum()),  # Fixed: Convert to int
                    "null_percentage": float((col_data.isnull().sum() / len(col_data)) * 100),  # Fixed: Convert to float
                    "unique_count": int(col_data.nunique()),  # Fixed: Convert to int
                    "unique_percentage": float((col_data.nunique() / len(col_data)) * 100)  # Fixed: Convert to float
                }
                
                if col_data.dtype in ['int64', 'float64']:
                    analysis.update({
                        "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,  # Fixed: Handle NaN
                        "median": float(col_data.median()) if not pd.isna(col_data.median()) else None,
                        "std": float(col_data.std()) if not pd.isna(col_data.std()) else None,
                        "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                        "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                        "skewness": float(col_data.skew()) if not pd.isna(col_data.skew()) else None,
                        "kurtosis": float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else None
                    })
                elif col_data.dtype == 'object':
                    try:
                        top_values = col_data.value_counts().head(5).to_dict()
                        analysis.update({
                            "top_values": top_values,
                            "avg_length": float(col_data.astype(str).str.len().mean()),
                            "max_length": int(col_data.astype(str).str.len().max())
                        })
                    except Exception:
                        analysis.update({
                            "top_values": {},
                            "avg_length": 0,
                            "max_length": 0
                        })
                
                column_analysis[column] = analysis
            
            # Use LLM to interpret column analysis
            prompt = f"""
            Analyze these column statistics and identify patterns, anomalies, and insights:
            
            {json.dumps(column_analysis, indent=2, default=str)}
            
            Focus on:
            1. Data quality issues
            2. Distribution patterns
            3. Potential relationships between columns
            4. Outliers or anomalies
            5. Business insights
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            column_analysis["llm_interpretation"] = response.content
            
            state["column_analysis"] = column_analysis
            state["current_step"] = "column_analyzer"
            
        except Exception as e:
            logger.error(f"Error in column analysis: {str(e)}")
            if "error_messages" not in state:
                state["error_messages"] = []
            if "column_analysis" not in state:
                state["column_analysis"] = {}
            state["error_messages"].append(f"Column analysis error: {str(e)}")
            
        return state
    
    def _generate_insights(self, state: AnalysisState) -> AnalysisState:
        """Generate insights from the data analysis"""
        logger.info("Generating insights...")
        
        try:
            df = state["dataset"]
            dataset_info = state["dataset_info"]
            
            # Ensure required keys exist in dataset_info
            if "numeric_columns" not in dataset_info:
                dataset_info["numeric_columns"] = df.select_dtypes(include=[np.number]).columns.tolist()
            if "categorical_columns" not in dataset_info:
                dataset_info["categorical_columns"] = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Correlation analysis for numeric columns
            correlations = {}
            numeric_cols = dataset_info.get("numeric_columns", [])
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.7:  # Fixed: Check for NaN
                            high_correlations.append({
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": float(corr_val)  # Fixed: Convert to float
                            })
                correlations["high_correlations"] = high_correlations
            
            # Use LLM to generate comprehensive insights
            prompt = f"""
            Based on the dataset analysis, generate key insights and findings:
            
            Dataset Info: {json.dumps(dataset_info, indent=2, default=str)}
            High Correlations: {json.dumps(correlations, indent=2, default=str)}
            
            Generate 5-10 specific, actionable insights that would be valuable for business decision-making.
            Focus on trends, patterns, anomalies, and opportunities.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            insights = response.content.split('\n')
            insights = [insight.strip() for insight in insights if insight.strip()]
            
            state["insights"] = insights
            state["current_step"] = "insight_generator"
            
        except Exception as e:
            logger.error(f"Error in insight generation: {str(e)}")
            if "error_messages" not in state:
                state["error_messages"] = []
            if "insights" not in state:
                state["insights"] = []
            state["error_messages"].append(f"Insight generation error: {str(e)}")
            
        return state
    
    def _plan_visualizations(self, state: AnalysisState) -> AnalysisState:
        """Plan appropriate visualizations based on data characteristics"""
        logger.info("Planning visualizations...")
        
        try:
            dataset_info = state["dataset_info"]
            insights = state["insights"]
            
            # Ensure required keys exist
            if "numeric_columns" not in dataset_info:
                df = state["dataset"]
                dataset_info["numeric_columns"] = df.select_dtypes(include=[np.number]).columns.tolist()
                dataset_info["categorical_columns"] = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Use LLM to plan visualizations
            prompt = f"""
            Plan the most effective visualizations for this dataset:
            
            Dataset Info: {json.dumps(dataset_info, indent=2, default=str)}
            Key Insights: {insights}
            
            Suggest 5-8 different visualization types with:
            1. Chart type (histogram, scatter, bar, line, heatmap, etc.)
            2. Columns to use
            3. Purpose/insight to communicate
            4. Title and description
            
            Return as a JSON list with this structure:
            [
                {{
                    "type": "histogram",
                    "columns": ["column_name"],
                    "title": "Distribution of...",
                    "description": "Shows the...",
                    "purpose": "Understand distribution"
                }}
            ]
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            try:
                # Extract JSON from response
                json_start = response.content.find('[')
                json_end = response.content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    viz_plan = json.loads(response.content[json_start:json_end])
                else:
                    viz_plan = self._create_default_viz_plan(dataset_info)
            except Exception:
                # Fallback visualization plan
                viz_plan = self._create_default_viz_plan(dataset_info)
            
            state["visualizations"] = viz_plan
            state["current_step"] = "visualization_planner"
            
        except Exception as e:
            logger.error(f"Error in visualization planning: {str(e)}")
            if "error_messages" not in state:
                state["error_messages"] = []
            if "visualizations" not in state:
                state["visualizations"] = []
            state["error_messages"].append(f"Visualization planning error: {str(e)}")
            # Ensure we have dataset_info for fallback
            if "dataset_info" not in state:
                state["dataset_info"] = {}
            state["visualizations"] = self._create_default_viz_plan(state["dataset_info"])
            
        return state
    
    def _create_default_viz_plan(self, dataset_info: Dict) -> List[Dict]:
        """Create a default visualization plan"""
        viz_plan = []
        
        # Ensure keys exist with defaults
        numeric_columns = dataset_info.get("numeric_columns", [])
        categorical_columns = dataset_info.get("categorical_columns", [])
        
        # Distribution plots for numeric columns
        for col in numeric_columns[:3]:
            viz_plan.append({
                "type": "histogram",
                "columns": [col],
                "title": f"Distribution of {col}",
                "description": f"Shows the distribution pattern of {col}",
                "purpose": "Understand data distribution"
            })
        
        # Bar plots for categorical columns
        for col in categorical_columns[:2]:
            viz_plan.append({
                "type": "bar",
                "columns": [col],
                "title": f"Frequency of {col}",
                "description": f"Shows the frequency of different {col} values",
                "purpose": "Understand categorical distribution"
            })
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_columns) > 1:
            viz_plan.append({
                "type": "heatmap",
                "columns": numeric_columns,
                "title": "Correlation Matrix",
                "description": "Shows correlations between numeric variables",
                "purpose": "Identify relationships"
            })
        
        return viz_plan
    
    def _create_charts(self, state: AnalysisState) -> AnalysisState:
        """Create the planned visualizations"""
        logger.info("Creating charts...")
        
        try:
            df = state["dataset"]
            viz_plans = state["visualizations"]
            
            # Fixed: Use a working matplotlib style
            try:
                plt.style.use('default')  # Fixed: Use default instead of seaborn-v0_8
            except:
                pass  # If style fails, continue with default
            
            for i, viz in enumerate(viz_plans):
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if viz["type"] == "histogram":
                        col = viz["columns"][0]
                        if col in df.columns and df[col].dtype in ['int64', 'float64']:
                            df[col].dropna().hist(bins=30, ax=ax, alpha=0.7)  # Fixed: Drop NaN values
                            ax.set_title(viz["title"])
                            ax.set_xlabel(col)
                            ax.set_ylabel('Frequency')
                    
                    elif viz["type"] == "bar":
                        col = viz["columns"][0]
                        if col in df.columns:
                            value_counts = df[col].value_counts().head(10)
                            value_counts.plot(kind='bar', ax=ax)
                            ax.set_title(viz["title"])
                            ax.set_xlabel(col)
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=45)
                    
                    elif viz["type"] == "heatmap":
                        numeric_cols = [col for col in viz["columns"] if col in df.columns and df[col].dtype in ['int64', 'float64']]
                        if len(numeric_cols) > 1:
                            corr_matrix = df[numeric_cols].corr()
                            # Fixed: Use matplotlib imshow instead of seaborn
                            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                            ax.set_xticks(range(len(corr_matrix.columns)))
                            ax.set_yticks(range(len(corr_matrix.columns)))
                            ax.set_xticklabels(corr_matrix.columns, rotation=45)
                            ax.set_yticklabels(corr_matrix.columns)
                            ax.set_title(viz["title"])
                            plt.colorbar(im, ax=ax)
                    
                    elif viz["type"] == "scatter":
                        if len(viz["columns"]) >= 2:
                            col1, col2 = viz["columns"][0], viz["columns"][1]
                            if col1 in df.columns and col2 in df.columns:
                                clean_data = df[[col1, col2]].dropna()  # Fixed: Remove NaN values
                                ax.scatter(clean_data[col1], clean_data[col2], alpha=0.6)
                                ax.set_xlabel(col1)
                                ax.set_ylabel(col2)
                                ax.set_title(viz["title"])
                    
                    plt.tight_layout()
                    plt.savefig(f'chart_{i+1}_{viz["type"]}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Failed to create {viz['type']} chart: {str(e)}")
                    plt.close()  # Fixed: Ensure figure is closed even on error
                    continue
            
            state["current_step"] = "chart_creator"
            
        except Exception as e:
            logger.error(f"Error in chart creation: {str(e)}")
            if "error_messages" not in state:
                state["error_messages"] = []
            state["error_messages"].append(f"Chart creation error: {str(e)}")
            
        return state
    
    def _generate_recommendations(self, state: AnalysisState) -> AnalysisState:
        """Generate actionable recommendations based on analysis"""
        logger.info("Generating recommendations...")
        
        try:
            insights = state["insights"]
            dataset_info = state["dataset_info"]
            
            # Use LLM to generate recommendations
            prompt = f"""
            Based on the complete data analysis, generate specific, actionable recommendations:
            
            Dataset Info: {json.dumps(dataset_info, indent=2, default=str)}
            Key Insights: {insights}
            
            Generate 5-10 specific recommendations that include:
            1. Data quality improvements
            2. Business opportunities
            3. Further analysis suggestions
            4. Action items for stakeholders
            
            Make recommendations specific, measurable, and actionable.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            recommendations = response.content.split('\n')
            recommendations = [rec.strip() for rec in recommendations if rec.strip()]
            
            state["recommendations"] = recommendations
            state["current_step"] = "recommendation_engine"
            
        except Exception as e:
            logger.error(f"Error in recommendation generation: {str(e)}")
            if "error_messages" not in state:
                state["error_messages"] = []
            if "recommendations" not in state:
                state["recommendations"] = []
            state["error_messages"].append(f"Recommendation generation error: {str(e)}")
            
        return state
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Main method to analyze a dataset"""
        logger.info(f"Starting analysis of dataset: {dataset_path}")
        
        try:
            # Load dataset
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
            
            # Initialize state with all required fields
            initial_state = AnalysisState(
                dataset=df,
                dataset_info={},
                column_analysis={},
                insights=[],
                visualizations=[],
                recommendations=[],
                current_step="",
                error_messages=[]
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare results
            results = {
                "dataset_info": final_state.get("dataset_info", {}),
                "column_analysis": final_state.get("column_analysis", {}),
                "insights": final_state.get("insights", []),
                "visualizations": final_state.get("visualizations", []),
                "recommendations": final_state.get("recommendations", []),
                "analysis_timestamp": datetime.now().isoformat(),
                "errors": final_state.get("error_messages", [])
            }
            
            # Generate summary report
            self._generate_report(results, dataset_path)
            
            logger.info("Analysis completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in dataset analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_report(self, results: Dict[str, Any], dataset_path: str):
        """Generate a comprehensive analysis report"""
        try:
            report_content = f"""
# Data Analysis Report
## Dataset: {dataset_path}
## Analysis Date: {results['analysis_timestamp']}

### Dataset Overview
- Shape: {results['dataset_info'].get('shape', 'N/A')}
- Columns: {len(results['dataset_info'].get('columns', []))}
- Missing Values: {sum(results['dataset_info'].get('null_counts', {}).values())}
- Duplicate Rows: {results['dataset_info'].get('duplicate_rows', 'N/A')}

### Key Insights
"""
            
            for i, insight in enumerate(results.get('insights', []), 1):
                report_content += f"{i}. {insight}\n"
            
            report_content += "\n### Recommendations\n"
            for i, rec in enumerate(results.get('recommendations', []), 1):
                report_content += f"{i}. {rec}\n"
            
            # Save report
            with open('analysis_report.md', 'w') as f:
                f.write(report_content)
            
            print("Analysis report saved as 'analysis_report.md'")
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

# Usage example and configuration
class DataAnalysisConfig:
    """Configuration class for easy customization"""
    
    def __init__(self):
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        self.model_name = "llama3-70b-8192"  # Fixed: Use correct model name
        self.output_directory = "analysis_output"
        self.chart_style = "default"  # Fixed: Use default style
        
    def validate(self):
        """Validate configuration"""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

def main():
    """Main function to run the data analysis system"""
    
    # Example usage
    config = DataAnalysisConfig()
    
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set the GROQ_API_KEY environment variable")
        return
    
    # Initialize the agent
    agent = DataAnalysisAgent(
        groq_api_key=config.groq_api_key,
        model_name=config.model_name
    )
    
    # Example: Analyze a dataset
    dataset_path = "your_dataset.csv"  # Replace with your dataset path
    
    if os.path.exists(dataset_path):
        results = agent.analyze_dataset(dataset_path)
        
        if "error" not in results:
            print("Analysis completed successfully!")
            print(f"Generated {len(results['insights'])} insights")
            print(f"Created {len(results['visualizations'])} visualizations")
            print(f"Provided {len(results['recommendations'])} recommendations")
        else:
            print(f"Analysis failed: {results['error']}")
    else:
        print(f"Dataset file not found: {dataset_path}")
        print("Please provide a valid dataset path")

if __name__ == "__main__":
    main()