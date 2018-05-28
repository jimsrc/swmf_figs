#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
import shared.funcs as sf


class cutter__3d_cut(object):
    """
    manipulate data && build 3D plots
    """
    def __init__(self):
        """
        First thing to do is build the parser
        """
        self.help = """
        Module to make a 3d-cut plot; two cuts: one in the r coordinate and other in phi).
        """
        self.parser = parser = argparse.ArgumentParser(
        description="""this gral description...""", 
        add_help=False
        )
        parser.add_argument(
        '-clim', '--clim',
        type=float,
        nargs=2,
        default=[None, None],
        help='colorbar limits',
        )
        parser.add_argument(
        '-cs', '--cb_scale',
        type=str,
        default='log', # 'log'
        help='colorbar scale ("linear" or "log")',
        )
        parser.add_argument(
        '-ro', '--ro',
        type=float,
        default=5.0,
        help='radius for the spherical shell to be plotted.',
        )
        parser.add_argument(
        '-pho', '--pho',
        type=float,
        default=0.0,
        help='longitude value for the cut',
        )
        parser.add_argument(
        '-dlon', '--dlon',
        type=float,
        default=0.0,
        help='interval width for the cut in longitude',
        )
        parser.add_argument(
        '-rr', '--r_range',
        type=float,
        nargs=2,
        default=[2., 5.],
        help='radius for the spherical shell to be plotted.',
        )
        parser.add_argument(
        '-v', '--verbose',
        type=str,
        default='debug',
        help='verbosity level (debug=minimal, info=extended)',
        )
        parser.add_argument(
        '-pazim', '--pazim', #action='store',
        type=float,
        default=-60.,
        help='perspective azimuth',
        )
        parser.add_argument(
        '-dpi', '--dpi', #action='store',
        type=int,
        default=100,
        help='dots per inch for savefig()',
        )

    def run(self, pa, **kws):
        custom_data = kws['custom_data']

        #--- build figs
        for finp in kws['finp_list_proc']:
            if not pa.fname_inp:    # massive mode
                fname_fig = finp.replace('.h5','__'+pa.vname+'.png')
                if pa.dir_dst:
                    # change the dir path
                    fname_fig = pa.dir_dst + '/' + fname_fig.split('/')[-1]
            else:                   # individual mode
                fname_fig = pa.fname_fig
        
            sf.make_3dplot(
                finp, 
                fname_fig, 
                pa.clim,
                vnames = custom_data.vnames, 
                data_processor = getattr(custom_data, 'process_'+pa.vname),
                verbose = pa.verbose, 
                ro = pa.ro,
                pho = pa.pho,
                r_range = pa.r_range,
                pazim = pa.pazim,
                cscale = pa.cb_scale,         # colorbar scale
                dpi = pa.dpi,
                wtimelabel = True,
                figsize = kws.get('figsize', getattr(pa, 'figsize', (7.2,4.8))),
            )

class cutter__r_cut(object):
    """
    manipulate data && build 2D plots
    """
    def __init__(self):
        """
        First thing to do is build the parser
        """
        self.help = """
        Module to make a 2d-cut plot; a cut in the r coordinate.
        """
        self.parser = parser = argparse.ArgumentParser(
        description="""this gral description...""", 
        add_help=False
        )
        parser.add_argument(
        '-clim', '--clim',
        type=float,
        nargs=2,
        default=[None, None],
        help='colorbar limits',
        )
        parser.add_argument(
        '-dlon', '--dlon',
        type=float,
        default=[None,None], #[0., 360.],
        nargs=2,
        metavar=('LON1','LON2'),
        help='interval width in longitude, in degrees.',
        )
        parser.add_argument(
        '-dth', '--dth',
        type=float,
        default=[None,None], #[-90., 90.],
        nargs=2,
        metavar=('TH1','TH2'),
        help='interval width in theta (co-latitude?), in degrees.',
        )
        parser.add_argument(
        '-ro', '--ro',
        type=float,
        default=5.0,
        help='radius for the spherical shell to be plotted.',
        )
        parser.add_argument(
        '-cs', '--cb_scale',
        type=str,
        default='log',
        help='colorbar scale ("linear" or "log")',
        )
        parser.add_argument(
        '-cl', '--cb_label',
        type=str,
        default='|B| [G]',
        help='colorbar label (e.g. variable name and units)',
        )
        parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        default=False,
        help="""If used, shows an interactive IPython plot; otherwise,
        it creates a .png figure.""",
        )
        parser.add_argument(
        '-v', '--verbose',
        type=str,
        default='debug',
        help='verbosity level (debug=minimal, info=extended)',
        )
        parser.add_argument(
        '-figsize', '--figsize',
        type=float,
        default=[6,4],
        nargs=2,
        metavar=('WIDTH','HEIGTH'),
        help='figure size',
        )

    def run(self, pa, **kws):
        custom_data = kws['custom_data']

        #--- build figs
        for finp in kws['finp_list_proc']:
            if not pa.fname_inp:    # massive mode
                fname_fig = finp.replace('.h5','__'+pa.vname+'.png')
                if pa.dir_dst:
                    # change the dir path
                    fname_fig = pa.dir_dst + '/' + fname_fig.split('/')[-1]
            else:                   # individual mode
                fname_fig = pa.fname_fig

            o = sf.r_cut(
                finp, 
                fname_fig, 
                ro=pa.ro, 
                dph=pa.dlon,
                dth=pa.dth,
                figsize=kws.get('figsize', getattr(pa, 'figsize', (6,4))),
                verbose=pa.verbose, 
                vnames=custom_data.vnames, 
                data_processor=getattr(custom_data, 'process_'+pa.vname),
                cscale=pa.cb_scale,                   # color scale
                colormap=kws.get('colormap', getattr(pa, 'colormap', 'gray')),
                cb_label=pa.cb_label,
                interactive=pa.interactive,
                wtimelabel = True,
            )
#EOF
