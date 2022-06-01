#!/usr/bin/env python3

import os
import subprocess
import sys
import re
from typing import Any
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import gh_post_comment, GitHubPR


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Rebase PR into branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def rebase_onto(pr: GitHubPR, repo: GitRepo, dry_run: bool = False, stable: bool = False) -> None:
    branch = f"pull/{pr.pr_num}/head"
    onto_branch = "refs/remotes/origin/viable/strict" if stable else pr.default_branch()
    remote_url = f"https://github.com/{pr.info['headRepository']['nameWithOwner']}.git"
    refspec = f"{branch}:{pr.head_ref()}"

    repo.fetch(branch, branch)
    repo._run_git("rebase", onto_branch, branch)
    if dry_run:
        push_result = repo._run_git("push", "--dry-run", "-f", remote_url, refspec)
    else:
        push_result = repo._run_git("push", "-f", remote_url, refspec)
    if "Everything up-to-date" in push_result:
        gh_post_comment(pr.org, pr.project, pr.pr_num,
                        f"Tried to rebase and push PR #{pr.pr_num}, but it was already up to date", dry_run=dry_run)
    else:
        gh_post_comment(pr.org, pr.project, pr.pr_num,
                        f"Successfully rebased `{pr.head_ref()}` onto `{onto_branch}`, please pull locally " +
                        f"before adding more changes (for example, via `git checkout {pr.head_ref()} && " +
                        "git pull --rebase`)", dry_run=dry_run)


def rebase_ghstack_onto(pr: GitHubPR, repo: GitRepo, dry_run: bool = False, stable: bool = False) -> None:
    if subprocess.run([sys.executable, "-m", "ghstack", "--help"], capture_output=True).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "ghstack"])
    orig_ref = f"{re.sub(r'/head$', '/orig', pr.head_ref())}"
    onto_branch = "refs/remotes/origin/viable/strict" if stable else pr.default_branch()

    repo.fetch(orig_ref, orig_ref)
    repo._run_git("rebase", onto_branch, orig_ref)

    os.environ["OAUTH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    with open('.ghstackrc', 'w+') as f:
        f.write('[ghstack]\n' +
                "github_url=github.com\n" +
                "github_username=pytorchmergebot\n" +
                "remote_name=origin")

    if dry_run:
        print("Don't know how to dry-run ghstack")
    else:
        ghstack_result = subprocess.run(["ghstack"], capture_output=True)
        push_result = ghstack_result.stdout.decode("utf-8")
        print(push_result)
        if ghstack_result.returncode != 0:
            raise Exception(f"\n```{push_result}```")
        # The contents of a successful push result should look like:
        # Summary of changes (ghstack 0.6.0)

        #  - Updated https://github.com/clee2000/random-testing/pull/2
        #  - Updated https://github.com/clee2000/random-testing/pull/1

        # Facebook employees can import your changes by running
        # (on a Facebook machine):

        #     ghimport -s https://github.com/clee2000/random-testing/pull/2

        # If you want to work on this diff stack on another machine:

        #     ghstack checkout https://github.com/clee2000/random-testing/pull/2
        org, project = repo.gh_owner_and_name()
        for line in push_result.splitlines():
            if "Updated" in line:
                pr_num = int(line.split("/")[-1])
                if pr_num != pr.pr_num:
                    gh_post_comment(pr.org, pr.project, pr_num,
                                    f"Rebased `{orig_ref}` onto `{onto_branch}` because #{pr.pr_num} was rebased, "
                                    "please pull locally before adding more changes (for example, via `git checkout "
                                    f"{orig_ref} && git pull --rebase`)", dry_run=dry_run)
                else:
                    gh_post_comment(pr.org, pr.project, pr_num,
                                    f"Successfully rebased `{orig_ref}` onto `{onto_branch}`, please pull locally " +
                                    f"before adding more changes (for example, via `git checkout {orig_ref} && " +
                                    "git pull --rebase`)", dry_run=dry_run)

        if f"Skipped https://github.com/{org}/{project}/pull/{pr.pr_num}" in push_result:
            gh_post_comment(pr.org, pr.project, pr.pr_num,
                            f"Tried to rebase and push PR #{pr.pr_num}, but it was already up to date", dry_run=dry_run)


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)

    if pr.is_closed():
        gh_post_comment(org, project, args.pr_num, f"PR #{args.pr_num} is closed, won't rebase", dry_run=args.dry_run)
        return

    try:
        if pr.is_ghstack_pr():
            rebase_ghstack_onto(pr, repo, dry_run=args.dry_run, stable=args.stable)
            return
        rebase_onto(pr, repo, dry_run=args.dry_run, stable=args.stable)
    except Exception as e:
        msg = f"Rebase failed due to {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
