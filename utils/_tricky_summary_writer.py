from typing import Any, Dict, List, Optional, Union

import torch._C
import torch.utils.tensorboard
import torch.utils.tensorboard.summary


# pylint: disable=protected-access, c-extension-no-member
class TrickySummaryWriter(torch.utils.tensorboard.SummaryWriter):

    # https://discuss.pytorch.org/t/how-to-add-graphs-to-hparams-in-tensorboard/109349/2
    def add_hparams(
        self,
        hparam_dict: Dict[str, Union[bool, str, float, int, None]],
        metric_dict: Dict[str, Union[float, int]],
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
    ):
        torch._C._log_api_usage_once('tensorboard.logging.add_hparams')
        if not (isinstance(hparam_dict, dict) and
                isinstance(metric_dict, dict)):
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = torch.utils.tensorboard.summary.hparams(
            hparam_dict, metric_dict, hparam_domain_discrete)

        # if not run_name:
        #     run_name = str(time.time())
        # logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        # with torch.utils.tensorboard.SummaryWriter(log_dir=logdir) as w_hp:
        #     w_hp.file_writer.add_summary(exp)
        #     w_hp.file_writer.add_summary(ssi)
        #     w_hp.file_writer.add_summary(sei)
        #     for k, v in metric_dict.items():
        #         w_hp.add_scalar(k, v)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)
