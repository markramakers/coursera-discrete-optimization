import solara
from dataclasses import dataclass
from collections import namedtuple
import math
from pathlib import Path
from pprint import pprint
import plotly.express as px
import pandas as pd

# Declare reactive variables at the top level. Components using these variables
# will be re-executed when their values change.
problem_selected = solara.reactive("")

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])
Problem = namedtuple("Problem", ['name', 'facilities', 'customers'])


def parseInputFile(input_data_file, filename: str) -> Problem:
    input_data = input_data_file.read()
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    return Problem(filename, facilities, customers)



def load_data():
    from pathlib import Path
    problems = {}
    for path in (Path.cwd() /  "data").iterdir():
        with open(path, 'r') as input_data_file:
            print(input_data_file)
            problems[path.name] = parseInputFile(input_data_file, path.name)
    pprint(problems.keys())
    return problems

problems = load_data()

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


@solara.component
def Page():
    # Calculate word_count within the component to ensure re-execution when reactive variables change.
    problem = problems.get(problem_selected.value)

    if problem:
        df_facilities = pd.DataFrame.from_records(problem.facilities)
        df_facilities['x'] = df_facilities[3].apply(lambda point: point.x)
        df_facilities['y'] = df_facilities[3].apply(lambda point: point.y)
        df_facilities['type'] = 'Facility'

        df_customers = pd.DataFrame.from_records(problem.customers)
        df_customers['x'] = df_customers[2].apply(lambda point: point.x)
        df_customers['y'] = df_customers[2].apply(lambda point: point.y)
        df_customers['type'] = 'Customer'

        df_map = pd.concat([df_facilities[['x', 'y', 'type']], df_customers[['x', 'y', 'type']]])
    with solara.Column() as main:
        with solara.Columns([1,3,5]):
            with solara.Column():
                solara.Button(label=f"Load data", on_click=load_data)
                solara.Select(label="Problems", values=[name for name in problems.keys()], value=problem_selected)

            with solara.Column():
                if problem:
                    solara.Markdown(f'''
                        # Problem: {problem.name if problem else ""}
                            - {len(problem.facilities)} Facilities
                            - {len(problem.customers)} Customers
                        
                        ''')
                    solara.Markdown("## Facilities")
                    solara.DataFrame(df_facilities,items_per_page=5)
                    solara.Markdown("## Customers")
                    solara.DataFrame(df_customers, items_per_page=5)
            with solara.Column():
                if problem:
                    fig = px.scatter(df_map, "x", "y", color="type", title="Map", height=800)
                    fig.update_traces(marker={'size': 4})

                    solara.FigurePlotly(fig)
                    

    return main




