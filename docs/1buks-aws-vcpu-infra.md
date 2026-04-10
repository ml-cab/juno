** How to use 1$ AWS Free Tier credits for 95 hours of fun! **

`g4dn.xlarge` is Nvidia's 4 vCPUs, 16 GiB, for tests lets take 2 of them.

The AWS Free Plan (the one from July 2025) explicitly restricts high-performance instances. High-spec instance types like `g4dn.xlarge` are not eligible for the free plan. 

So that we need to request some quotas!

```
aws service-quotas request-service-quota-increase   --service-code ec2   --quota-code L-DB2E81BA   --desired-value 12   --region eu-north-1
```

verify Quota:

```
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region eu-north-1 --query "RequestedQuotas[?QuotaCode=='L-DB2E81BA'].[Status,DesiredValue,Created]" --output table
```

outputs:

```
-------------------------------------------------------------
|          ListRequestedServiceQuotaChangeHistory           |
+--------------+-------+------------------------------------+
|  CASE_OPENED |  12.0 |  2026-04-02T01:56:51.160000+03:00  |
+--------------+-------+------------------------------------+
```


Standard free plan is limited from accessing a subset of AWS services and offerings that would immediately consume the entire Free Tier credit amount. GPU instances fall into exactly that category.

1. Go to **AWS Billing and Cost Management Console** → `https://console.aws.amazon.com/billing/`
2. Click **"Upgrade Plan"** — it's in the navigation bar or the Cost and Usage widget on the home dash-board
3. Confirm the upgrade

When you upgrade to paid plan, your remaining Free Tier credits will automatically apply to future AWS bills until they expire. So your $100 will still be there and will cover the g4dn.xlarge costs — you're not giving anything up, you're just unlocking GPU access.

One heads-up on budget: 2× g4dn.xlarge at $0.526/hr burns roughly **$1.05/hr**. Your $100 credit gives you ~95 hours of runtime, so consider using stop/start workflow to not waste it.


