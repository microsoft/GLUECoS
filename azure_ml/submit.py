#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import textwrap
import ruamel.yaml as yaml
import argparse
import tempfile
import json
import datetime

from azureml.core import Workspace, Experiment, Datastore, get_run
from azureml.core.compute import ComputeTarget
from azureml.train.estimator import Estimator

entry_script_content = """
    import os
    import sys
    import argparse
    import re

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--command", required=True, help="the command to run")

    args, unknown = parser.parse_known_args()

    command = args.command + " "
    command = re.sub(r'\\\\;', ';', command)

    for i in range(len(unknown)):
        x = unknown[i]
        if x == "--OUTPUT_DIR":
            output_dir = unknown[i + 1]
    for var in os.environ:
        os.environ[var] = os.path.expandvars(os.environ[var])
    os.makedirs(os.environ['OUTPUT_DIR'], exist_ok=True)

    os.system("nvidia-smi")
    # os.chdir(os.environ['CODE'])
    os.system("cp -r {} {}".format(os.environ['CODE'], os.path.join(os.environ['HOME'], 'code')))
    os.chdir(os.path.join(os.environ['HOME'], 'code'))
    print("Executing {}".format(command))
    return_code = os.system(command)
    sys.exit(return_code >> 8)

"""


def pretty_print(list_to_print):
    n_cols = len(list_to_print[0])
    max_l = [max(len(t[i]) for t in list_to_print) for i in range(n_cols)]
    format_string = ['{:<' + str(m) + '}' for m in max_l[:-1]]
    format_string.append('{}')
    format_string = ' '.join(format_string)
    for i in range(len(list_to_print)):
        print(format_string.format(*list_to_print[i]))


def list_exp(args):
    log = read_log_file()

    experiments = log['experiments']
    print("List of experiments")
    to_print = [["Experiment", "LastModified"]]
    experiments_ordered = sorted(
        experiments, key=lambda exp: log['experiments'][exp]['modified'])
    for exp in experiments_ordered:
        t = datetime.datetime.fromtimestamp(
            int(log['experiments'][exp]['modified']))
        to_print.append([exp, str(t)])
    pretty_print(to_print)


def status(args):
    ws = get_workspace()
    log = read_log_file()
    if log is None:
        print("No log files found. Exiting!!")
        exit(0)

    experiments = log['experiments']
    if args.experiment_name in experiments:
        exp = Experiment(ws, name=log['aml_experiment_name'])
        print("Experiment: {}".format(args.experiment_name))
        print(experiments[args.experiment_name]['description'])
        all = True
        if args.j:
            all = False
            to_print = [["Name", "Status", "Link"]]
            # print("Name", "Status", "Link", sep='\t')
        else:
            to_print = [["Name", "Status"]]
            # print("Name", "Status", sep='\t')

        for run, name in experiments[args.experiment_name]['ids']:
            if all:
                run = get_run(exp, run)
                details = run.get_details()
                to_print.append([run.tags['name'], details['status']])
                # print(run.tags['name'], details['status'], sep='\t')
            elif name in args.j:
                run = get_run(exp, run)
                details = run.get_details()
                to_print.append(
                    [run.tags['name'], details['status'], run.get_portal_url()])
                # print(run.tags['name'], details['status'], run.get_portal_url(), sep='\t')
                to_print.append(["", "", ""])
        pretty_print(to_print)
    else:
        print("Experiment not found")


def read_log_file():
    if os.path.exists(os.path.join('.azureml', 'log')):
        with open(os.path.join('.azureml', 'log'), 'r') as f:
            log = json.load(f)
        return log
    else:
        print("Log file doesn't exist, which means that the experiment wasn't initialized")
        exit(0)


def write_log_file(log):
    with open(os.path.join('.azureml', 'log'), 'w') as f:
        json.dump(log, f)


def get_workspace():
    ws = Workspace.from_config()
    return ws


def run(args):
    with open(args.run_spec_file, "r") as f:
        run_spec = yaml.load(f, Loader=yaml.SafeLoader)

    log = read_log_file()

    ws = get_workspace()
    experiment = Experiment(workspace=ws, name=log['aml_experiment_name'])

    experiments = log['experiments']

    # Checking if experiment with same name already exists and cancelling and deleting it if needed
    if args.experiment_name in experiments:
        print("Experiment already exists. Please give a different name")
        exit(0)

    submitted_runs = []
    all = True
    if args.j:
        all = False

    source_directory = tempfile.TemporaryDirectory()
    entry_script_file = "entry.py"

    with open(os.path.join(source_directory.name, entry_script_file), "w") as f:
        f.write(textwrap.dedent(entry_script_content).strip() + "\n")

    script_params = {}
    environment_variables = {}
    for x in run_spec['volumes']:
        if 'path' in x:
            script_params["--{}".format(x['name'])] = Datastore(
                workspace=ws, name=x['datastore']).path(x['path']).as_mount()
            environment_variables[x['name']] = str(
                script_params["--{}".format(x['name'])])
        if x['name'] == 'OUTPUT_DIR':
            output_dir_datastore = x['datastore']

    if 'environment_variables' in run_spec:
        for x in run_spec['environment_variables']:
            environment_variables[x['name']] = x['value']

    setup_command = ""
    if 'setup' in run_spec:
        for x in run_spec['setup']:
            setup_command += x
            setup_command += '; '

    compute_target = ComputeTarget(workspace=ws, name=run_spec['compute_name'])
    description = run_spec['description']

    rtype = 'run'
    for run in run_spec['runs']:
        if all or run['name'] in args.j:
            script_params["--OUTPUT_DIR"] = Datastore(workspace=ws, name=output_dir_datastore).path(
                "Experiments/{}/{}/{}".format(log['aml_experiment_name'], args.experiment_name, run['name'])).as_mount()
            environment_variables['OUTPUT_DIR'] = str(
                script_params["--OUTPUT_DIR"])
            command = setup_command + run['command']
            script_params['--command'] = command

            params = {
                'use_gpu': True,
                'custom_docker_image': run_spec['docker_image'],
                'user_managed': True,
                'source_directory': source_directory.name,
                'entry_script': entry_script_file,
                'script_params': script_params,
                'environment_variables': environment_variables,
                'compute_target': compute_target,
            }

            est = Estimator(**params)

            tags = {'name': run['name'], 'experiment_name': args.experiment_name}
            submitted_run = experiment.submit(est, tags=tags)
            print("Submitting ", tags['name'], submitted_run.get_portal_url())

            submitted_runs.append([submitted_run.id, run['name']])

    log['experiments'][args.experiment_name] = {'type': rtype,
                                                'ids': submitted_runs,
                                                'modified': datetime.datetime.now().timestamp(),
                                                'output_dir_datastore': output_dir_datastore,
                                                'description': description}
    write_log_file(log)

    source_directory.cleanup()


def init_workspace(args):
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace_name = args.workspace_name

    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    print("Workspace configuration succeeded")

    ws.write_config()

    experiment_name = 'GLUECoS'
    experiment = Experiment(ws, name=experiment_name)
    log = {'aml_experiment_name': experiment_name, 'experiments': {}}
    write_log_file(log)
    print("Creating new experiment")


def main():
    parser = argparse.ArgumentParser("GLUECoS Run Submitter")
    subparsers = parser.add_subparsers(dest='cmd')

    run_parser = subparsers.add_parser("run")
    list_parser = subparsers.add_parser("list")
    status_parser = subparsers.add_parser("status")
    init_parser = subparsers.add_parser("init")

    run_parser.add_argument("run_spec_file", type=str, help="spec file containing runs")
    run_parser.add_argument("experiment_name", type=str, help='name of experiment')
    run_parser.add_argument("-j", type=str, nargs='+', help='list of jobs to run')
    run_parser.set_defaults(func=run)

    status_parser.set_defaults(func=status)
    status_parser.add_argument("experiment_name", type=str, help='list jobs in experiment')
    status_parser.add_argument("-j", type=str, nargs='+', help='list of jobs to show info on')

    list_parser.set_defaults(func=list_exp)

    init_parser.set_defaults(func=init_workspace)
    init_parser.add_argument("--subscription_id", type=str, required=True, help='ID of subscription containing workspace')
    init_parser.add_argument("--resource_group", type=str, required=True, help='resource group containing AML workspace')
    init_parser.add_argument("--workspace_name", type=str, required=True, help='AML workspace name')

    args = parser.parse_args()
    if args.cmd:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
