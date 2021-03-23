import logging
import sys
import time
import os
from pathlib import Path

from numpy import linspace, mean, ceil

import gnpy.core.ansi_escapes as ansi_escapes
import gnpy.core.exceptions as exceptions
from gnpy.core.elements import Transceiver, Fiber, RamanFiber
from gnpy.core.equipment import trx_mode_params
from gnpy.core.network import build_network
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.core.utils import db2lin, lin2db,automatic_nch
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import load_equipment, load_network, load_json, save_network, load_requests,requests_from_json,disjunctions_from_json
from gnpy.topology.request import (ResultElement, jsontocsv, compute_path_dsjctn, requests_aggregation,
                                   BLOCKING_NOPATH, correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction,
                                   PathRequest, compute_constrained_path, propagate)
from gnpy.tools.cli_examples import load_common_data
from gnpy.tools.demand_to_services import DemandToServices
from gnpy.tools.plots import plot_baseline, plot_results


_logger = logging.getLogger(__name__)


class RunPathRequests:
    """

    """

    def __init__(self, equipment_filename, topology_filename, hecson_demand_filename, service_filename, simulation_filename, save_raw_network_filename,bw_list):
        """

        :param equipment_filename:
        :param topology_filename:
        :param simulation_filename:
        :param save_raw_network_filename:
        """

        try:
            self.equipment_filename = Path(equipment_filename)
            self.topology = Path(topology_filename)
            self.equipment = load_equipment(equipment_filename)
            self.network = load_network(self.topology,self.equipment)
            self.demands_to_services = DemandToServices(hecson_demand_filename,service_filename, bw_list)
            self.serviceFile = Path(self.demands_to_services.convert_to_services("json", True))
            self.save_network_before_autodesign = save_raw_network_filename
            if save_raw_network_filename is not None:
                save_network(self.network, save_raw_network_filename)
                _logger.info(f'{ansi_escapes.blue}Raw network (no optimizations) saved to {save_raw_network_filename}{ansi_escapes.reset}')

            self.sim_params = SimParams(**load_json(simulation_filename)) if simulation_filename is not None else None
            if not self.sim_params:
                if next((node for node in self.network if isinstance(node, RamanFiber)), None) is not None:
                    _logger.info(
                        f'{ansi_escapes.red}Invocation error:{ansi_escapes.reset} 'f'RamanFiber requires passing simulation params via --sim-params')  # print changed to log
                    sys.exit(1)
            else:
                Simulation.set_params(self.sim_params)
            self.transceivers = {n.uid: n for n in self.network if isinstance(n, Transceiver)}

            if not self.transceivers:
                sys.exit('Network has no transceivers!')
            if len(self.transceivers) < 2:
                sys.exit('Network has only one transceiver!')

        except exceptions.EquipmentConfigError as e:
            _logger.info(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}')
            sys.exit(1)

        except exceptions.NetworkTopologyError as e:
            _logger.info(
                f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)
        except exceptions.ConfigurationError as e:
            _logger.info(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)
        except exceptions.ParametersError as e:
            _logger.info(
                f'{ansi_escapes.red}Simulation parameters error:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)
        except exceptions.ServiceError as e:
            _logger.info(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)

    def transmission_main_example(self, args_source, args_destination, args_power, mode, chBW, args_plot=False,
                                  args_list_nodes=False
                                  , args_save_network=None, args_show_channels=False, args_verbose=0):
        """

        :param args_source:
        :param args_destination:
        :param args_power:
        :param mode:
        :param chBW:
        :param args_plot:
        :param args_list_nodes:
        :param args_save_network:
        :param args_show_channels:
        :param args_verbose:
        :return:
        """

        if args_plot:
            plot_baseline(self.network)

        if args_list_nodes:
            for uid in self.transceivers:
                _logger.info(uid)  # print changed to log
            sys.exit()
        # First try to find exact match if source/destination provided
        if args_source:
            source = self.transceivers[args_source]
            # source = self.transceivers.pop(args_source, None)
            valid_source = True if source else False
        else:
            _logger.error('No source node specified!')
            sys.exit()

        if args_destination:
            # destination = self.transceivers.pop(args_destination, None)
            destination = self.transceivers[args_destination]
            valid_destination = True if destination else False
        else:

            _logger.error('No destination node specified: picking random transceiver')
            sys.exit()

        # TODO: Removed unnecssary code checks assuming input data is given correctly, otherwise exit

        _logger.info(f'source = {args_source!r}')
        _logger.info(f'destination = {args_destination!r}')

        params = {}
        params['request_id'] = 0
        params['trx_type'] = ''
        params['trx_mode'] = mode
        params['source'] = source.uid
        params['destination'] = destination.uid
        params['bidir'] = True  # TODO: Should be true (?!)
        params['nodes_list'] = [destination.uid]
        params['loose_list'] = ['strict']
        params['format'] = ''
        params['path_bandwidth'] = 0
        params['spacing'] = chBW * 1e9
        # params['nbchannels'] = 0
        trx_params = trx_mode_params(self.equipment, trx_type_variety='Adva', trx_mode=mode)
        if args_power:
            trx_params['power'] = db2lin(float(args_power)) * 1e-3
        params.update(trx_params)
        req = PathRequest(**params)

        power_mode = self.equipment['Span']['default'].power_mode
        _logger.info('\n'.join([f'Power mode is set to {power_mode}',
                                f'=> it can be modified in eqpt_config.json - Span']))  # print changed to log

        pref_ch_db = lin2db(req.power * 1e3)  # reference channel power / span (SL=20dB)
        pref_total_db = pref_ch_db + lin2db(req.nb_channel)  # reference total power / span (SL=20dB)
        try:
            build_network(self.network, self.equipment, pref_ch_db, pref_total_db)
        except exceptions.NetworkTopologyError as e:
            _logger.info(
                f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)
        except exceptions.ConfigurationError as e:
            _logger.info(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')  # print changed to log
            sys.exit(1)
        # TODO: HERE IS THE SHORTEST PATH CALCULATION
        path = compute_constrained_path(self.network, req)

        spans = [s.params.length for s in path if isinstance(s, RamanFiber) or isinstance(s, Fiber)]
        _logger.info(f'\nThere are {len(spans)} fiber spans over {sum(spans) / 1000:.0f} km between {source.uid} '
                     f'and {destination.uid}')  # print changed to log
        _logger.info(f'\nNow propagating between {source.uid} and {destination.uid}:')  # print changed to log

        try:
            p_start, p_stop, p_step = self.equipment['SI']['default'].power_range_db
            p_num = abs(int(round((p_stop - p_start) / p_step))) + 1 if p_step != 0 else 1
            power_range = list(linspace(p_start, p_stop, p_num))
        except TypeError:
            _logger.info(
                'invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')  # print changed to log
            power_range = [0]

        if not power_mode:
            # power cannot be changed in gain mode
            power_range = [0]
        for dp_db in power_range:
            req.power = db2lin(pref_ch_db + dp_db) * 1e-3
            if power_mode:
                _logger.info(
                    f'\nPropagating with input power = {ansi_escapes.cyan}{lin2db(req.power * 1e3):.2f} dBm{ansi_escapes.reset}:')  # print changed to log
            else:
                _logger.info(
                    f'\nPropagating in {ansi_escapes.cyan}gain mode{ansi_escapes.reset}: power cannot be set manually')  # print changed to log
            infos = propagate(path, req, self.equipment)
            if len(power_range) == 1:
                # for elem in path:
                # print(elem)
                if power_mode:
                    _logger.info(
                        f'\nTransmission result for input power = {lin2db(req.power * 1e3):.2f} dBm:')  # print changed to log
                else:
                    _logger.info(f'\nTransmission results:')  # print changed to log
                _logger.info(
                    f'  Final SNR total (0.1 nm): {ansi_escapes.cyan}{mean(destination.snr_01nm):.02f} dB{ansi_escapes.reset}')  # print changed to log

            else:
                print(path[-1])  # _logger.info changed to log

        if args_save_network is not None:
            save_network(self.network, args_save_network)
            _logger.info(
                f'{ansi_escapes.blue}Network (after autodesign) saved to {args_save_network}{ansi_escapes.reset}')  # print changed to log

        if args_show_channels:
            _logger.info('\nThe total SNR per channel at the end of the line is:')  # print changed to log
            _logger.info(
                '{:>5}{:>26}{:>26}{:>28}{:>28}{:>28}'.format(
                    'Ch. #',
                    'Channel frequency (THz)',
                    'Channel power (dBm)',
                    'OSNR ASE (signal bw, dB)',
                    'SNR NLI (signal bw, dB)',
                    'SNR total (signal bw, dB)'))  # print changed to log
            for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(
                    infos.carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr):
                ch_freq = final_carrier.frequency * 1e-12
                ch_power = lin2db(final_carrier.power.signal * 1e3)
                _logger.info(
                    '{:5}{:26.2f}{:26.2f}{:28.2f}{:28.2f}{:28.2f}'.format(
                        final_carrier.channel_number, round(
                            ch_freq, 2), round(
                            ch_power, 2), round(
                            ch_osnr, 2), round(
                            ch_snr_nl, 2), round(
                            ch_snr, 2)))  # print changed to log

        if not args_source:
            _logger.info(f'\n(No source node specified: picked {source.uid})')  # print changed to log
        elif not valid_source:
            _logger.info(f'\n(Invalid source node {args_source!r} replaced with {source.uid})')  # print changed to log

        if not args_destination:
            _logger.info(f'\n(No destination node specified: picked {destination.uid})')  # print changed to log
        elif not valid_destination:
            _logger.info(
                f'\n(Invalid destination node {args_destination!r} replaced with {destination.uid})')  # print changed to log

        if args_plot:
            plot_results(self.network, path, source, destination)
        if len(power_range) == 1:
            return mean(destination.snr_01nm)
        else:
            return 0

    def path_requests_run(self):

        _logger.info(f'Computing path requests {self.serviceFile} into JSON format')
        print(
            f'{ansi_escapes.blue}Computing path requests {os.path.relpath(self.serviceFile)} into JSON format{ansi_escapes.reset}')

        (equipment, network) = load_common_data(self.equipment_filename, self.topology, self.sim_params,
                                                Path(self.save_network_before_autodesign))

        # Build the network once using the default power defined in SI in eqpt config
        # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
        # spacing, f_min and f_max
        p_db = equipment['SI']['default'].power_dbm

        p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                                 equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
        try:
            build_network(network, equipment, p_db, p_total_db)
        except exceptions.NetworkTopologyError as e:
            print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
            sys.exit(1)
        except exceptions.ConfigurationError as e:
            print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
            sys.exit(1)
        if self.save_network_before_autodesign is not None:
            save_network(network, self.save_network_before_autodesign)
            print(f'{ansi_escapes.blue}Network (after autodesign) saved to { self.save_network_before_autodesign}{ansi_escapes.reset}')
        oms_list = build_oms_list(network, equipment)

        try:
            data = load_requests(self.serviceFile, equipment, bidir=True,
                                 network=network, network_filename=self.topology)
            rqs = requests_from_json(data, equipment)
        except exceptions.ServiceError as e:
            print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
            sys.exit(1)
        # check that request ids are unique. Non unique ids, may
        # mess the computation: better to stop the computation
        all_ids = [r.request_id for r in rqs]
        if len(all_ids) != len(set(all_ids)):
            for item in list(set(all_ids)):
                all_ids.remove(item)
            msg = f'Requests id {all_ids} are not unique'
            _logger.critical(msg)
            sys.exit()
        rqs = correct_json_route_list(network, rqs)

        # pths = compute_path(network, equipment, rqs)
        dsjn = disjunctions_from_json(data)

        print(f'{ansi_escapes.blue}List of disjunctions{ansi_escapes.reset}')
        print(dsjn)
        # need to warn or correct in case of wrong disjunction form
        # disjunction must not be repeated with same or different ids
        dsjn = deduplicate_disjunctions(dsjn)

        # Aggregate demands with same exact constraints
        print(f'{ansi_escapes.blue}Aggregating similar requests{ansi_escapes.reset}')

        rqs, dsjn = requests_aggregation(rqs, dsjn)
        # TODO export novel set of aggregated demands in a json file

        print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
        print(rqs)

        print(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
        try:
            pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
        except exceptions.DisjunctionError as this_e:
            print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
            sys.exit(1)

        print(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
        propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs,
                                                                                               pths)
        # Note that deepcopy used in compute_path_with_disjunction returns
        # a list of nodes which are not belonging to network (they are copies of the node objects).
        # so there can not be propagation on these nodes.

        pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

        print(f'{ansi_escapes.blue}Result summary{ansi_escapes.reset}')
        header = ['req id', '  demand', '  snr@bandwidth A-Z (Z-A)', '  snr@0.1nm A-Z (Z-A)',
                  '  Receiver minOSNR', '  mode', '  Gbit/s', '  nb of tsp pairs',
                  'N,M or blocking reason']
        data = []
        data.append(header)
        total_no = len(propagatedpths)
        blocked = 0
        for i, this_p in enumerate(propagatedpths):
            rev_pth = reversed_propagatedpths[i]
            if rev_pth and this_p:
                psnrb = f'{round(mean(this_p[-1].snr), 2)} ({round(mean(rev_pth[-1].snr), 2)})'
                psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' + \
                       f' ({round(mean(rev_pth[-1].snr_01nm), 2)})'
            elif this_p:
                psnrb = f'{round(mean(this_p[-1].snr), 2)}'
                psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}'

            try:
                if rqs[i].blocking_reason in BLOCKING_NOPATH:

                    line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} :',
                            f'-', f'-', f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                            f'-', f'{rqs[i].blocking_reason}']
                else:
                    if rqs[i].blocking_reason == "NO_SPECTRUM":
                        blocked += 1
                    line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                            psnr, f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                            f'-', f'{rqs[i].blocking_reason}']
            except AttributeError:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                        psnr, f'{rqs[i].OSNR + equipment["SI"]["default"].sys_margins}',
                        f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate)}', f'({rqs[i].N},{rqs[i].M})']
            data.append(line)

        col_width = max(len(word) for row in data for word in row[2:])  # padding
        firstcol_width = max(len(row[0]) for row in data)  # padding
        secondcol_width = max(len(row[1]) for row in data)  # padding
        for row in data:
            firstcol = ''.join(row[0].ljust(firstcol_width))
            secondcol = ''.join(row[1].ljust(secondcol_width))
            remainingcols = ''.join(word.center(col_width, ' ') for word in row[2:])
            print(f'{firstcol} {secondcol} {remainingcols}')

        print("Total Services Planned: "+ str(total_no))
        print("Services Blocked: "+ str(blocked))
        bp = (blocked/total_no)*100
        print("Blocking Probability: "+str(bp)+" %")
        print(
            f'{ansi_escapes.yellow}Result summary shows mean SNR and OSNR (average over all channels){ansi_escapes.reset}')




if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    snr_gnpy = RunPathRequests(r"C:\Users\SaiP\Documents\gitProject\oopt-gnpy\gnpy\example-data\eqpt_config.json",
                               r"C:\Users\SaiP\Documents\gitProject\oopt-gnpy\gnpy\example-data\germany17.xls",
                               r"C:\Users\SaiP\Documents\lrzgitProject\org.opticon.optirecon\build\resources\main\nwData"
                               r"\Demands_Germany_17_updated.json", r"germany17_services.json", None
                               , r"C:\Users\SaiP\Documents\gitProject\oopt-gnpy\gnpy\example-data\germany17_save_nw.json", [37.5, 50.0, 75.0, 87.5])

    snr_gnpy.path_requests_run()
    end_time = time.time()
    diff = end_time - start_time

    print("Time taken for 136 demands and 272 services: "+str(diff)+ " sec")
    # transmission_main_example(args_source='Berlin',args_destination='Dortmund', args_power=-1.2)