import pandas as pd
import numpy as np
from scipy import stats
import re
import traceback
from typing import Dict, List, Any, Tuple, Union
import sys
from functools import reduce
import requests
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import re
import traceback
from typing import Dict, List, Any, Tuple, Union
import sys
from functools import reduce
import requests
from datetime import datetime
import os

class ExpressionEvaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_mapping = {}
        
        # Pattern for matching column names (case insensitive)
        self.column_pattern = r'([Qq]\d+[Xx_]\d+[_]?\d*)'
        
        # Statistical operations
        self.statistical_ops = {
            'MEAN': np.mean,
            'MEDIAN': np.median,
            'MODE': lambda x: stats.mode(x, keepdims=False).mode,
            'STD': np.std,
            'VAR': np.var,
            'MIN': np.min,
            'MAX': np.max,
            'SUM': np.sum,
            'COUNT': len,
            'Q1': lambda x: np.percentile(x, 25),
            'Q3': lambda x: np.percentile(x, 75),
            'IQR': lambda x: np.percentile(x, 75) - np.percentile(x, 25)
        }
        
        # Comparison operators
        self.comparison_ops = {
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '>': np.greater,
            '<': np.less,
            '==': np.equal,
            '!=': np.not_equal,
            '=': np.equal
        }

    def evaluate_with_na_handling(self, expression: str, df: pd.DataFrame) -> pd.Series:
        """Evaluate expression after handling NAs for specific columns."""
        columns = re.findall(self.column_pattern, expression, re.IGNORECASE)
        mask = pd.Series(True, index=df.index)
        for col in columns:
            col_name = self.get_column_case_insensitive(col)
            mask &= ~df[col_name].isna()
        
        filtered_df = df[mask].copy()
        evaluator = ExpressionEvaluator(filtered_df)
        result = evaluator.evaluate_expression(expression)
        full_result = pd.Series(0, index=df.index)
        full_result[result.index] = result
        return full_result

    def get_column_case_insensitive(self, col: str) -> str:
        """Find column name ignoring case, extra spaces, and potential variations."""
        col = col.strip()
        print(f"\n🔎 Searching for column: '{col}'")
        
        # Create mapping of stripped column names
        stripped_cols = {c.strip().upper(): c for c in self.df.columns}
        
        # Try exact match after stripping
        col_upper = col.upper()
        if col_upper in stripped_cols:
            print(f"✅ Exact match found: {stripped_cols[col_upper]}")
            return stripped_cols[col_upper]
            
        # Try replacing X/x variations
        col_normalized = col_upper.replace('X', '_')
        for orig_col, df_col in stripped_cols.items():
            if orig_col.replace('X', '_') == col_normalized:
                print(f"✅ Found column through normalization: {df_col}")
                return df_col
        
        print(f"❌ Column '{col}' not found! Available columns:")
        print(self.df.columns.tolist())
        raise ValueError(f"Column not found: {col}")

    def parse_condition(self, condition: str) -> Tuple[str, str, float]:
        """Parse a single condition like 'Q40_1_1>=6' into (column, operator, value)"""
        pattern = r'(Q\d+[_x]\d+(?:_\d+)?)\s*([><=]+)\s*(\d+(?:\.\d*)?)'
        match = re.match(pattern, condition, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid condition format: {condition}")
        return match.groups()

    def evaluate_condition(self, condition: str) -> pd.Series:
        """Evaluate a single condition like 'Q40_1_1>=6' with null handling"""
        col, op, val = self.parse_condition(condition)
        col_name = self.get_column_case_insensitive(col)
        
        # Create mask for non-null values
        mask = ~self.df[col_name].isna()
        
        # Create result series initialized with zeros
        result = pd.Series(0, index=self.df.index)
        
        # Apply comparison only where we have valid data
        valid_data = self.df[col_name][mask]
        result[mask] = self.apply_comparison(valid_data, op, float(val))
        
        return result

    def evaluate_complex_condition(self, expression: str) -> pd.Series:
        """Handle complex conditions with null value handling"""
        print(f"Evaluating complex condition: {expression}")
        
        # Get all columns involved
        columns = re.findall(self.column_pattern, expression, re.IGNORECASE)
        
        # Create mask for non-null values
        mask = pd.Series(True, index=self.df.index)
        for col in columns:
            col_name = self.get_column_case_insensitive(col)
            mask &= ~self.df[col_name].isna()
        
        # For expressions like Q40_1_1>=6-Q40_1_3>=6>0
        if re.search(r'>=\d+-.*>=\d+[><=]\d+', expression):
            parts = expression.split('-')
            first_condition = parts[0].strip()
            second_part = parts[1].strip()
            
            # Extract the final comparison
            match = re.match(r'(.*?)([><=]+)(\d+)$', second_part)
            if match:
                second_condition = match.group(1).strip()
                final_op = match.group(2)
                final_val = float(match.group(3))
                
                # Evaluate both conditions
                first_result = self.evaluate_condition(first_condition)
                second_result = self.evaluate_condition(second_condition)
                
                # Convert to integers (0 or 1)
                first_result = first_result.astype(int)
                second_result = second_result.astype(int)
                
                # Calculate difference
                result = first_result - second_result
                
                # Apply final comparison
                final_result = self.apply_comparison(result, final_op, final_val)
                return final_result.astype(int) & mask
        
        # For other complex conditions
        parts = re.split(r'([+\-*/])', expression)
        parts = [p.strip() for p in parts if p.strip()]
        
        # Process each part
        results = []
        operators = []
        
        for part in parts:
            if part in ['+', '-', '*', '/']:
                operators.append(part)
                continue
                
            try:
                result = self.evaluate_condition(part)
                results.append(result & mask)  # Only consider where we have valid data
            except Exception as e:
                print(f"Error processing part {part}: {str(e)}")
                raise
        
        # Combine results
        final_result = results[0]
        for i, op in enumerate(operators):
            if op == '+':
                final_result = final_result.astype(int) + results[i + 1].astype(int)
            elif op == '-':
                final_result = final_result.astype(int) - results[i + 1].astype(int)
            elif op == '*':
                final_result = final_result.astype(int) * results[i + 1].astype(int)
            elif op == '/':
                final_result = final_result.astype(float) / results[i + 1].replace(0, np.nan)
        
        # Check for final comparison if not already handled
        if not re.search(r'>=\d+-.*>=\d+[><=]\d+', expression):
            final_comp = re.search(r'([><=]+)(\d+)$', expression)
            if final_comp:
                op, val = final_comp.groups()
                final_result = self.apply_comparison(final_result, op, float(val))
        
        return final_result.astype(int)
    def evaluate_arithmetic_expression(self, expression: str) -> pd.Series:
        """Evaluate arithmetic expressions between columns with improved parentheses handling."""
        print(f"Evaluating arithmetic expression: {expression}")
        
        def evaluate_part(expr: str) -> pd.Series:
            """Recursively evaluate parts of the expression."""
            expr = expr.strip()
            
            # Handle parentheses first
            if expr.startswith('('):
                # Find matching closing parenthesis
                count = 1
                pos = 1
                while pos < len(expr) and count > 0:
                    if expr[pos] == '(':
                        count += 1
                    elif expr[pos] == ')':
                        count -= 1
                    pos += 1
                
                if count != 0:
                    raise ValueError(f"Unmatched parentheses in expression: {expr}")
                
                inner_expr = expr[1:pos-1]
                if pos < len(expr):
                    rest = expr[pos:].strip()
                    if rest:
                        inner_result = evaluate_part(inner_expr)
                        if rest.startswith('+'):
                            return inner_result + evaluate_part(rest[1:])
                        elif rest.startswith('-'):
                            return inner_result - evaluate_part(rest[1:])
                        elif rest.startswith('*'):
                            return inner_result * evaluate_part(rest[1:])
                        elif rest.startswith('/'):
                            return inner_result / evaluate_part(rest[1:]).replace(0, np.nan)
                return evaluate_part(inner_expr)
            
            # Split by operators for non-parentheses expressions
            parts = re.split(r'([+\-*/])', expr)
            parts = [p.strip() for p in parts if p.strip()]
            
            if not parts:
                raise ValueError(f"Empty expression: {expr}")
            
            # Process each part
            result = None
            current_op = '+'
            
            for part in parts:
                if part in ['+', '-', '*', '/']:
                    current_op = part
                    continue
                    
                # Get value for this part
                if re.match(self.column_pattern, part, re.IGNORECASE):
                    col_name = self.get_column_case_insensitive(part)
                    value = self.df[col_name].fillna(0)
                else:
                    try:
                        value = float(part)
                    except ValueError:
                        # If conversion fails, try to evaluate it as a sub-expression
                        value = evaluate_part(part)
                
                # Apply operation
                if result is None:
                    result = value
                else:
                    if current_op == '+':
                        result = result + value
                    elif current_op == '-':
                        result = result - value
                    elif current_op == '*':
                        result = result * value
                    elif current_op == '/':
                        result = result / value.replace(0, np.nan)
            
            return result
        
        try:
            # Get all columns involved
            columns = re.findall(self.column_pattern, expression, re.IGNORECASE)
            
            # Create mask for non-null values
            mask = pd.Series(True, index=self.df.index)
            for col in columns:
                col_name = self.get_column_case_insensitive(col)
                mask &= ~self.df[col_name].isna()
            
            # Evaluate the expression recursively
            result = evaluate_part(expression)
            
            # Apply null value handling
            result[~mask] = np.nan
            
            return result.fillna(0)
            
        except Exception as e:
            print(f"Error in arithmetic expression evaluation: {str(e)}")
            raise

    def evaluate_percentage_calculation(self, expression: str) -> pd.Series:
        """Evaluate percentage calculations with better nested parentheses handling."""
        print(f"Evaluating percentage calculation: {expression}")
        
        # Find comparison operator and threshold
        for op in sorted(self.comparison_ops.keys(), key=len, reverse=True):
            if op in expression:
                parts = expression.split(op)
                if len(parts) == 2 and '%' in parts[1]:
                    expr = parts[0].strip()
                    threshold = float(parts[1].rstrip('%'))
                    
                    # Handle nested parentheses
                    def find_matching_parenthesis(s: str, start: int) -> int:
                        """Find the matching closing parenthesis."""
                        count = 1
                        i = start + 1
                        while i < len(s):
                            if s[i] == '(':
                                count += 1
                            elif s[i] == ')':
                                count -= 1
                                if count == 0:
                                    return i
                            i += 1
                        return -1

                    def extract_expression_parts(expr: str) -> tuple:
                        """Extract numerator and denominator from expression with nested parentheses."""
                        expr = expr.strip()
                        if not expr.startswith('('):
                            # Simple division
                            num, denom = expr.split('/')
                            return num.strip(), denom.strip()
                        
                        # Find the end of numerator parentheses
                        num_end = find_matching_parenthesis(expr, 0)
                        if num_end == -1:
                            raise ValueError("Unmatched parentheses in numerator")
                        
                        # Extract numerator
                        numerator = expr[1:num_end].strip()
                        
                        # Find denominator
                        rest = expr[num_end+1:].strip()
                        if not rest.startswith('/'):
                            raise ValueError("Invalid expression format")
                        
                        denominator = rest[1:].strip()
                        return numerator, denominator

                    try:
                        num_expr, denom_expr = extract_expression_parts(expr)
                        print(f"Numerator: {num_expr}")
                        print(f"Denominator: {denom_expr}")
                        
                        # Get columns involved in calculation
                        num_cols = re.findall(self.column_pattern, num_expr, re.IGNORECASE)
                        denom_cols = re.findall(self.column_pattern, denom_expr, re.IGNORECASE)
                        
                        # Create mask for non-null values
                        mask = pd.Series(True, index=self.df.index)
                        for col in num_cols + denom_cols:
                            col_name = self.get_column_case_insensitive(col)
                            mask &= ~self.df[col_name].isna()
                        
                        # Evaluate expressions only where we have valid data
                        numerator = self.evaluate_arithmetic_expression(num_expr)
                        denominator = self.evaluate_arithmetic_expression(denom_expr)
                        
                        # Calculate percentage where we have valid data, else NaN
                        result = pd.Series(np.nan, index=self.df.index)
                        valid_denom = (denominator != 0) & mask
                        result[valid_denom] = (numerator[valid_denom] / denominator[valid_denom]) * 100
                        
                        # Fill NaN with 0 and apply comparison
                        result = result.fillna(0)
                        return self.apply_comparison(result, op, threshold)
                        
                    except Exception as e:
                        print(f"Error processing percentage calculation: {str(e)}")
                        raise
        
        raise ValueError("Invalid percentage expression")

    def evaluate_logical_operation(self, expression: str) -> pd.Series:
        """Evaluate logical operations (AND/OR) with improved null handling."""
        print(f"Evaluating logical operation: {expression}")
        
        # Normalize the expression
        expression = re.sub(r'\s+or\s+', ' OR ', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\s+vs\s+', ' OR ', expression, flags=re.IGNORECASE)
        
        # Split by operators
        if ' AND ' in expression:
            parts = expression.split(' AND ')
            combine_op = 'AND'
        else:
            parts = expression.split(' OR ')
            combine_op = 'OR'
        
        # Get all columns involved
        columns = []
        for part in parts:
            cols = re.findall(self.column_pattern, part, re.IGNORECASE)
            columns.extend(cols)
        
        # Create mask for non-null values
        mask = pd.Series(True, index=self.df.index)
        for col in columns:
            col_name = self.get_column_case_insensitive(col)
            mask &= ~self.df[col_name].isna()
        
        # Evaluate each part
        results = []
        for part in parts:
            part = part.strip()
            try:
                if '%' in part:
                    result = self.evaluate_percentage_calculation(part)
                else:
                    result = self.evaluate_condition(part)
                results.append(result & mask)  # Only consider where we have valid data
            except Exception as e:
                print(f"Error evaluating part '{part}': {str(e)}")
                raise
        
        # Combine results
        if combine_op == 'AND':
            return reduce(lambda x, y: x & y, results)
        else:
            return reduce(lambda x, y: x | y, results)

    def evaluate_statistical_comparison(self, expression: str) -> pd.Series:
        """Evaluate statistical comparisons with null handling."""
        match = re.match(r'([^><=]+)\s*([><=]+)\s*(\w+)\(([^)]+)\)', expression)
        if not match:
            raise ValueError(f"Invalid statistical comparison format: {expression}")
        
        col, op, func, stat_col = match.groups()
        func = func.upper()
        if func not in self.statistical_ops:
            raise ValueError(f"Unknown statistical function: {func}")
        
        col_name = self.get_column_case_insensitive(col)
        stat_col_name = self.get_column_case_insensitive(stat_col)
        
        # Create mask for non-null values
        mask = ~self.df[col_name].isna() & ~self.df[stat_col_name].isna()
        
        # Calculate statistic only on non-null values
        stat_value = self.statistical_ops[func](self.df[stat_col_name][mask].fillna(0))
        
        # Apply comparison only where we have valid data
        result = pd.Series(0, index=self.df.index)
        result[mask] = self.apply_comparison(self.df[col_name][mask], op, stat_value)
        
        return result

    def apply_comparison(self, values: pd.Series, operator: str, threshold: float) -> pd.Series:
        """Apply comparison operation."""
        print(f"\nApplying comparison: {operator} {threshold}")
        
        if operator not in self.comparison_ops:
            operator = '==' if operator == '=' else operator
            
        result = self.comparison_ops[operator](values, threshold)
        return result.astype(int)

    def evaluate_expression(self, expression: str) -> pd.Series:
        """Main evaluation function with improved null handling."""
        try:
            expression = expression.strip()
            print(f"\n🔍 Evaluating expression: '{expression}'")
            
            # Handle each type of expression
            if '%' in expression:
                return self.evaluate_percentage_calculation(expression)
                
            if any(op in expression.upper() for op in ['AND', 'OR', 'VS']):
                return self.evaluate_logical_operation(expression)
                
            if any(f"{op}(" in expression.upper() for op in self.statistical_ops.keys()):
                return self.evaluate_statistical_comparison(expression)
            
            if any(f"{op}(" in expression for op in ['MIN', 'MAX', 'AVG']):
                return self.evaluate_statistical_comparison(expression)
            
            if re.search(r'[><=]+\d+[-+*/]', expression):
                return self.evaluate_complex_condition(expression)
            
            if any(op in expression for op in ['+', '-', '*', '/']):
                # Check if it's arithmetic with comparison
                for op in sorted(self.comparison_ops.keys(), key=len, reverse=True):
                    if op in expression:
                        parts = expression.split(op)
                        if len(parts) == 2:
                            left = self.evaluate_arithmetic_expression(parts[0])
                            right = float(parts[1].rstrip('%'))
                            return self.apply_comparison(left, op, right)
                return self.evaluate_arithmetic_expression(expression)
            
            # Simple comparison
            for op in sorted(self.comparison_ops.keys(), key=len, reverse=True):
                if op in expression:
                    parts = expression.split(op)
                    if len(parts) == 2:
                        col_name = self.get_column_case_insensitive(parts[0].strip())
                        threshold = float(parts[1].strip())
                        
                        # Handle null values
                        mask = ~self.df[col_name].isna()
                        result = pd.Series(0, index=self.df.index)
                        result[mask] = self.apply_comparison(self.df[col_name][mask], op, threshold)
                        return result
            
            # Simple column reference
            col_name = self.get_column_case_insensitive(expression)
            # Handle null values for simple column reference
            return self.df[col_name].fillna(0)
            
        except Exception as e:
            print(f"⚠️ Error evaluating expression: {str(e)}")
            traceback.print_exc()
            return pd.Series(0, index=self.df.index)

def process_segments(df: pd.DataFrame, segment_rules: Dict[str, str]) -> pd.DataFrame:
    """Process segments according to rules with improved null handling."""
    print("\nProcessing segments...")
    result_df = df.copy()
    print(f"Total rows in dataset: {len(result_df)}")
    
    # Process each segment
    for segment_col, rule in segment_rules.items():
        print(f"\nProcessing {segment_col}...")
        
        if pd.isna(rule):
            print(f"Skipping {segment_col} - rule is null")
            continue
            
        print(f"Rule: {rule}")
        
        try:
            # Create evaluator
            evaluator = ExpressionEvaluator(result_df)
            
            # Use the new NA handling evaluation
            result = evaluator.evaluate_with_na_handling(rule, result_df)
            
            # Store results
            result_df[f'{segment_col}_result'] = result
            
            # Print results
            value_counts = result_df[f'{segment_col}_result'].value_counts()
            print(f"Results for {segment_col}:")
            print(value_counts)
            
            # Get matching NPIs
            if 'NPI ' in result_df.columns:
                matching_mask = (result_df[f'{segment_col}_result'] == 1)
                matching_npis = result_df.loc[matching_mask, 'NPI '].tolist()
                print(f"\nNPI Results for {segment_col}:")
                print(f"Total matching NPIs: {len(matching_npis)}")
                if matching_npis:
                    print("First 5 matching NPIs:", matching_npis[:5])
            
        except Exception as e:
            print(f"Error in segment processing: {str(e)}")
            traceback.print_exc()
            result_df[f'{segment_col}_result'] = 0
            continue
    
    return result_df

def main(raw_data_path: str, segment_rules_path: str) -> pd.DataFrame:
    """Main function to process segmentation with rules from database."""
    try:
        print("Reading raw data...")
        # Read CSV and handle column names
        df = pd.read_csv(raw_data_path)
        
        # Print all column names for debugging
        print("\nOriginal column names:")
        print(df.columns.tolist())
        
        # Clean column names - remove trailing/leading spaces and store original mapping
        original_cols = df.columns.tolist()
        clean_cols = [col.strip() for col in df.columns]
        col_mapping = dict(zip(clean_cols, original_cols))
        
        print("\nColumn mapping:")
        for clean, orig in col_mapping.items():
            print(f"{clean} -> {orig}")
            
        # Update DataFrame with clean column names
        df.columns = clean_cols
        print(f"\nLoaded {len(df)} rows from raw data")
        
        # Read segment rules from the segment rules table
        print("\nReading segment rules...")
        segment_df = pd.read_csv(segment_rules_path)
        
        # Create segment rules dictionary
        segment_rules = {}
        for _, row in segment_df.iterrows():
            if pd.notna(row['Segment_1']):  # Only process if segment rule exists
                segment_name = f"Segment_{row['Segment_Count']}"
                segment_rules[segment_name] = row['Segment_1']
        
        print("\nLoaded segment rules:")
        for segment, rule in segment_rules.items():
            print(f"{segment}: {rule}")
        
        # Process segments
        result_df = process_segments(df, segment_rules)
        
        # Save results
        output_path = "segmentation_results.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved complete results to: {output_path}")
        
        return result_df
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        raise
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        traceback.print_exc()
        raise

# if __name__ == "__main__":
#     try:
#         # File paths
#         raw_data_path = "https://parthenon.customerinsights.ai/ds/FFdRNn6l8DQOaBI"
#         segment_rules_path = "https://parthenon.customerinsights.ai/ds/DOoryx8crPRgcn4"
        
#         # Process segmentation
#         print("\n=== Starting Segmentation Processing ===\n")
#         result_df = main(raw_data_path, segment_rules_path)
        
#         # Read segment rules
#         segment_df = pd.read_csv(segment_rules_path)
#         segment_conditions = {}
        
#         # Store all segment conditions
#         for _, row in segment_df.iterrows():
#             for i in range(1, 4):  # Process Segment_1, Segment_2, Segment_3
#                 segment_col = f'Segment_{i}'
#                 if pd.notna(row[segment_col]):
#                     segment_name = f"Segment_{row['Segment_Count']}_result"
#                     segment_conditions[segment_name] = row[segment_col]
        
#         # Print summary
#         print("\n=== Processing Summary ===")
#         print(f"Total rows processed: {len(result_df)}")
        
#         # Initialize dictionary to store all segments' data
#         all_segments_data = {
#             "Process_DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         for col in result_df.columns:
#             if col.endswith('_result'):
#                 counts = result_df[col].value_counts()
#                 print(f"\n{col} distribution:")
#                 print(counts)
#                 if len(counts) > 0:
#                     print(f"Percentage True: {(counts.get(1, 0) / len(result_df)) * 100:.2f}%")
#                     if 'NPI' in result_df.columns:
#                         matching_npis = result_df.loc[result_df[col] == 1, 'NPI'].tolist()
#                         print(f"Total matching NPIs: {len(matching_npis)}")
#                         if matching_npis:
#                             print("matching NPIs:", matching_npis)
                            
#                             # Get the actual condition
#                             actual_condition = segment_conditions.get(col, "Unknown Condition")
#                             segment_number = col.split('_')[1]  # Get segment number (1, 2, or 3)
                            
#                             # Add data for this segment
#                             all_segments_data[f"Segment_{segment_number}_Condition"] = actual_condition
#                             all_segments_data[f"Segment_{segment_number}_NPIs"] = str(matching_npis)
#                             all_segments_data[f"Segment_{segment_number}_Count"] = len(matching_npis)
        
#         # Create the final JSON with all segments in one row
#         json = {
#             "data": [all_segments_data]
#         }
        
#         # Send single API request with all segments' data
#         response = requests.post(
#             'https://ciparthenon-api.azurewebsites.net/apiRequest?account=demo&route=data/826395/insert?api_version=2022.01',
#             json=json
#         )
#         print(f"API Response: {response.json()}")
        
#         print("\n=== Processing completed successfully ===")
        
#     except Exception as e:
#         print("\n=== Processing failed ===")
#         print(f"Error: {str(e)}")
#         traceback.print_exc()
#         sys.exit(1)
    
#     sys.exit(0)

app = Flask(__name__)

@app.route('/testing',methods=['GET'])

def hello():

    return "App is working"


@app.route('/process_segments', methods=['GET'])
def process_segments_api():
    try:
        # Get URLs from request body
        raw_data_path = "https://parthenon.customerinsights.ai/ds/FFdRNn6l8DQOaBI"
        segment_rules_path = "https://parthenon.customerinsights.ai/ds/DOoryx8crPRgcn4"

        # if not raw_data_path or not segment_rules_path:
        #     return jsonify({
        #         'status': 'error',
        #         'message': 'Both raw_data_path and segment_rules_path are required'
        #     }), 400

        # Process segmentation
        print("\n=== Starting Segmentation Processing ===\n")
        result_df = main(raw_data_path, segment_rules_path)
        
        # Read segment rules
        segment_df = pd.read_csv(segment_rules_path)
        segment_conditions = {}
        
        # Store all segment conditions
        for _, row in segment_df.iterrows():
            for i in range(1, 4):  # Process Segment_1, Segment_2, Segment_3
                segment_col = f'Segment_{i}'
                if pd.notna(row[segment_col]):
                    segment_name = f"Segment_{row['Segment_Count']}_result"
                    segment_conditions[segment_name] = row[segment_col]
        
        # Initialize dictionary to store all segments' data
        all_segments_data = {
            "Process_DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results = []
        for col in result_df.columns:
            if col.endswith('_result'):
                counts = result_df[col].value_counts()
                if len(counts) > 0:
                    matching_mask = (result_df[col] == 1)
                    matching_npis = result_df.loc[matching_mask, 'NPI'].tolist()
                    
                    if matching_npis:
                        # Get the actual condition
                        actual_condition = segment_conditions.get(col, "Unknown Condition")
                        segment_number = col.split('_')[1]
                        
                        # Add data for this segment
                        all_segments_data[f"Segment_{segment_number}_Condition"] = actual_condition
                        all_segments_data[f"Segment_{segment_number}_NPIs"] = str(matching_npis)
                        all_segments_data[f"Segment_{segment_number}_Count"] = len(matching_npis)
                        
                        results.append({
                            'segment': col,
                            'condition': actual_condition,
                            'count': len(matching_npis),
                            'npis': matching_npis,
                            'percentage': (len(matching_npis) / len(result_df)) * 100
                        })
        
        # Send data to API
        api_json = {
            "data": [all_segments_data]
        }
        
        response = requests.post(
            'https://ciparthenon-api.azurewebsites.net/apiRequest?account=demo&route=data/826395/insert?api_version=2022.01',
            json=api_json
        )
        
        # Return results
        return jsonify({
            'status': 'success',
            'message': 'Processing completed successfully',
            'results': results,
            'api_response': response.json()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)